#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <curl/curl.h>
#include <set>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable> // Для std::condition_variable - синхронизация потоков
#include <future>             // Для std::future - получение результатов из потоков

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t newLength = size * nmemb;
    try {
        data->append((char*)contents, newLength);
        return newLength;
    } catch (std::bad_alloc& e) {
        return 0;
    }
}

std::string make_get(const std::string& url) {
    CURL* curl;
    CURLcode res;
    std::string response_data;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-link-parser/1.0");
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    return response_data;
}

std::string getDomain(const std::string &url) {
    size_t start = url.find("://");
    if (start == std::string::npos) {
        start = 0;
    } else {
        start += 3;
    }
    size_t end = url.find("/", start);
    if (end == std::string::npos) {
        return url.substr(start);
    }
    return url.substr(start, end - start);
}

std::string normalizeUrl(const std::string &baseUrl, const std::string &link) {
    if (link.find("http://") == 0 || link.find("https://") == 0) {
        return link;
    } else if (link.find("//") == 0) {
        size_t protocol_end = baseUrl.find("://");
        if (protocol_end == std::string::npos) {
            return "https:" + link;
        }
        std::string protocol = baseUrl.substr(0, protocol_end);
        return protocol + ":" + link;
    } else if (link[0] == '/') {
        size_t protocol_end = baseUrl.find("://");
        if (protocol_end == std::string::npos) {
            return link;
        }
        size_t domain_end = baseUrl.find("/", protocol_end + 3);
        if (domain_end == std::string::npos) {
            return baseUrl + link;
        } else {
            return baseUrl.substr(0, domain_end) + link;
        }
    } else {
        std::string base = baseUrl;
        if (base.back() != '/') {
            size_t last_slash = base.find_last_of('/');
            if (last_slash > base.find("://") + 2) {
                base = base.substr(0, last_slash + 1);
            } else {
                base += '/';
            }
        }
        return base + link;
    }
}

std::vector<std::string> extractURLs(const std::string& url, const std::string& target_domain) {
    std::vector<std::string> links;
    std::string response_data = make_get(url);
    
    if (response_data.empty()) {
        return links;
    }

    // Improved regex to find various URL formats
    std::regex urlPattern(R"((https?:\/\/[^\s\'\"<>]+|(?:\/\/)[^\s\'\"<>]+|\/[^\s\'\"<>][^\s\'\"<>]*))", std::regex_constants::icase);
    
    auto words_begin = std::sregex_iterator(response_data.begin(), response_data.end(), urlPattern);
    auto words_end = std::sregex_iterator();
    
    std::set<std::string> unique_links;
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string raw_link = match.str();
        
        // Normalize the URL
        std::string absolute_url = normalizeUrl(url, raw_link);
        
        // Check if URL belongs to the target domain
        std::string link_domain = getDomain(absolute_url);
        if (link_domain == target_domain) {
            if (unique_links.find(absolute_url) == unique_links.end()) {
                unique_links.insert(absolute_url);
                links.push_back(absolute_url);
            }
        }
    }
    return links;
}

int main() {
    std::string start_url;
    
    std::cout << "Ведите URL: ";
    std::cin >> start_url;
    
    std::string target_domain = getDomain(start_url);
    std::cout << "Текущий домен domain: " << target_domain << std::endl;

    std::set<std::string> visited;
    std::queue<std::string> to_visit;
    std::vector<std::string> all_links;

    // Мьютекс для защиты общих структур данных (visited, to_visit, all_links)
    // Нужен, т.к. несколько потоков будут одновременно читать и писать в эти структуры
    std::mutex mtx;

    // Condition variable для уведомления о появлении новых URL в очереди
    // Потоки будут ждать на cv, когда очередь пуста, и просыпаться при добавлении URL
    std::condition_variable cv;

    // Счётчик активных задач - сколько потоков сейчас обрабатывают URL
    int active_tasks = 0;

    to_visit.push(start_url);
    visited.insert(start_url);
    all_links.push_back(start_url);

    int max_pages = 50;
    int processed_pages = 0;
    bool done = false;  // Флаг завершения работы

    // Определяем количество потоков
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    // Создаём рабочие потоки
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&]() {
            while (true) {
                std::string current_url;

                // Блок с захватом мьютекса для получения URL из очереди чтобы не анлочить вручную работу
                {
                    // unique_lock позволяет освобождать мьютекс при ожидании на cv
                    std::unique_lock<std::mutex> lock(mtx);

                    // Ждём пока появится URL в очереди ИЛИ работа завершена
                    // condition_variable::wait освобождает мьютекс на время ожидания
                    cv.wait(lock, [&]() {
                        return !to_visit.empty() || done;
                    });

                    // Если работа завершена и очередь пуста - выходим
                    if (done && to_visit.empty()) {
                        return;
                    }

                    // Проверяем лимит обработанных страниц
                    if (processed_pages >= max_pages) {
                        done = true;
                        cv.notify_all();  // Будим все потоки для завершения
                        return;
                    }

                    // Берём URL из очереди
                    current_url = to_visit.front();
                    to_visit.pop();
                    processed_pages++;
                    active_tasks++;  // Увеличиваем счётчик активных задач
                }

                std::cout << "Обработка: " << current_url << std::endl;

                // Скачиваем и парсим страницу (вне мьютекса - долгая операция)
                std::vector<std::string> new_links = extractURLs(current_url, target_domain);

                // Добавляем найденные ссылки в общие структуры
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    for (const std::string& link : new_links) {
                        if (visited.find(link) == visited.end()) {
                            visited.insert(link);
                            to_visit.push(link);
                            all_links.push_back(link);
                        }
                    }
                    active_tasks--;  // Уменьшаем счётчик активных задач

                    // Если очередь пуста и нет активных задач - завершаем
                    if (to_visit.empty() && active_tasks == 0) {
                        done = true;
                    }
                }

                // Уведомляем другие потоки о новых URL или завершении
                cv.notify_all();
            }
        });
    }

    // Ждём завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "\nНайдено " << all_links.size() << " уникальных ссылок для текущего домена " << target_domain << ":" << std::endl;
    for (const std::string& link : all_links) {
        std::cout << link << std::endl;
    }
    
    return 0;
}