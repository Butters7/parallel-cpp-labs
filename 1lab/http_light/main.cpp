#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <curl/curl.h>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <exception>

// Константы
const int MAX_PAGES = 50;
const size_t MAX_QUEUE_SIZE = 500;  // Back-pressure: ограничение очереди

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t newLength = size * nmemb;
    try {
        data->append((char*)contents, newLength);
        return newLength;
    } catch (std::bad_alloc&) {
        return 0;
    }
}

// RAII обёртка для CURL - предотвращает утечки ресурсов
class CurlHandle {
    CURL* curl;
public:
    CurlHandle() : curl(curl_easy_init()) {}
    ~CurlHandle() { if (curl) curl_easy_cleanup(curl); }
    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
    CURL* get() { return curl; }
    operator bool() const { return curl != nullptr; }
};

std::string make_get(const std::string& url) {
    std::string response_data;
    CurlHandle curl;

    if (curl) {
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "libcurl-link-parser/1.0");
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl.get());
        if (res != CURLE_OK) {
            std::cerr << "[ERROR] " << url << ": " << curl_easy_strerror(res) << std::endl;
            return "";
        }
    }
    return response_data;
}

std::string getDomain(const std::string& url) {
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

std::string normalizeUrl(const std::string& baseUrl, const std::string& link) {
    if (link.empty() || link[0] == '#' || link.find("javascript:") == 0) {
        return "";
    }

    if (link.find("http://") == 0 || link.find("https://") == 0) {
        return link;
    } else if (link.find("//") == 0) {
        size_t protocol_end = baseUrl.find("://");
        if (protocol_end == std::string::npos) {
            return "https:" + link;
        }
        return baseUrl.substr(0, protocol_end) + ":" + link;
    } else if (link[0] == '/') {
        size_t protocol_end = baseUrl.find("://");
        if (protocol_end == std::string::npos) {
            return link;
        }
        size_t domain_end = baseUrl.find("/", protocol_end + 3);
        if (domain_end == std::string::npos) {
            return baseUrl + link;
        }
        return baseUrl.substr(0, domain_end) + link;
    } else {
        std::string base = baseUrl;
        if (!base.empty() && base.back() != '/') {
            size_t last_slash = base.find_last_of('/');
            size_t protocol_pos = base.find("://");
            if (protocol_pos != std::string::npos && last_slash > protocol_pos + 2) {
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

    // Regex для href атрибутов
    std::regex hrefPattern(R"(href\s*=\s*["']([^"']+)["'])", std::regex_constants::icase);

    auto begin = std::sregex_iterator(response_data.begin(), response_data.end(), hrefPattern);
    auto end = std::sregex_iterator();

    std::set<std::string> unique_links;

    for (auto i = begin; i != end; ++i) {
        if ((*i).size() > 1) {
            std::string raw_link = (*i)[1].str();
            std::string absolute_url = normalizeUrl(url, raw_link);

            if (absolute_url.empty()) continue;

            std::string link_domain = getDomain(absolute_url);
            if (link_domain == target_domain) {
                if (unique_links.find(absolute_url) == unique_links.end()) {
                    unique_links.insert(absolute_url);
                    links.push_back(absolute_url);
                }
            }
        }
    }
    return links;
}

// Сохранение в XML
void saveToXML(const std::string& filename,
               const std::string& domain,
               const std::unordered_map<std::string, std::vector<std::string>>& site_map) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return;
    }

    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<sitemap domain=\"" << domain << "\">\n";

    for (const auto& entry : site_map) {
        file << "  <page url=\"" << entry.first << "\">\n";
        for (const auto& link : entry.second) {
            file << "    <link>" << link << "</link>\n";
        }
        file << "  </page>\n";
    }

    file << "</sitemap>\n";
    std::cout << "Карта сайта сохранена в: " << filename << std::endl;
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::string start_url;
    std::cout << "Введите URL: ";
    std::cin >> start_url;

    std::string target_domain = getDomain(start_url);
    std::cout << "Целевой домен: " << target_domain << std::endl;

    // Общие структуры данных
    std::unordered_set<std::string> visited;
    std::queue<std::string> to_visit;
    std::vector<std::string> all_links;
    std::unordered_map<std::string, std::vector<std::string>> site_map;

    // Синхронизация
    std::mutex mtx;
    std::mutex cout_mtx;  // Отдельный мьютекс для вывода
    std::condition_variable cv;
    std::atomic<int> active_tasks{0};
    std::atomic<int> processed_pages{0};
    std::atomic<bool> done{false};

    // Для обработки исключений
    std::mutex exc_mtx;
    std::exception_ptr thread_exception;

    to_visit.push(start_url);
    visited.insert(start_url);
    all_links.push_back(start_url);

    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<std::thread> threads;

    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&]() {
            try {
                while (true) {
                    std::string current_url;

                    // Критическая секция: взять задачу
                    {
                        std::unique_lock<std::mutex> lock(mtx);

                        // Ждём: есть работа ИЛИ завершение
                        cv.wait(lock, [&]() {
                            return !to_visit.empty() || done.load() ||
                                   (to_visit.empty() && active_tasks.load() == 0);
                        });

                        // Условие выхода: done И очередь пуста
                        if (to_visit.empty()) {
                            if (active_tasks.load() == 0) {
                                done.store(true);
                            }
                            if (done.load()) {
                                lock.unlock();
                                cv.notify_all();
                                return;
                            }
                            continue;  // Ждём дальше
                        }

                        // Проверяем лимит
                        if (processed_pages.load() >= MAX_PAGES) {
                            done.store(true);
                            lock.unlock();
                            cv.notify_all();
                            return;
                        }

                        current_url = to_visit.front();
                        to_visit.pop();
                        active_tasks.fetch_add(1);
                    }

                    // Обработка ВНЕ мьютекса (HTTP запрос)
                    int page_num = processed_pages.load() + 1;
                    {
                        std::lock_guard<std::mutex> cout_lock(cout_mtx);
                        std::cout << "[" << page_num << "/" << MAX_PAGES << "] " << current_url << std::endl;
                    }

                    std::vector<std::string> new_links = extractURLs(current_url, target_domain);

                    // Критическая секция: добавить результаты
                    {
                        std::lock_guard<std::mutex> lock(mtx);

                        // Сохраняем карту
                        site_map[current_url] = new_links;

                        // Добавляем новые ссылки (с back-pressure)
                        for (const auto& link : new_links) {
                            if (to_visit.size() >= MAX_QUEUE_SIZE) break;  // Back-pressure
                            if (visited.find(link) == visited.end()) {
                                visited.insert(link);
                                to_visit.push(link);
                                all_links.push_back(link);
                            }
                        }

                        processed_pages.fetch_add(1);
                        active_tasks.fetch_sub(1);

                        if (processed_pages.load() >= MAX_PAGES) {
                            done.store(true);
                        }
                    }

                    cv.notify_all();  // Уведомляем ПОСЛЕ освобождения мьютекса
                }
            } catch (...) {
                // Сохраняем исключение
                std::lock_guard<std::mutex> lock(exc_mtx);
                if (!thread_exception) {
                    thread_exception = std::current_exception();
                }
                done.store(true);
                cv.notify_all();
            }
        });
    }

    // Ждём завершения потоков
    for (auto& th : threads) {
        th.join();
    }

    // Проверяем исключения
    if (thread_exception) {
        std::rethrow_exception(thread_exception);
    }

    // Вывод результатов
    std::cout << "\n=== Результаты ===" << std::endl;
    std::cout << "Найдено " << all_links.size() << " уникальных ссылок для домена " << target_domain << std::endl;
    std::cout << "Обработано страниц: " << processed_pages.load() << std::endl;

    for (const auto& link : all_links) {
        std::cout << link << std::endl;
    }

    // Сохранение в XML
    saveToXML("sitemap.xml", target_domain, site_map);

    curl_global_cleanup();
    return 0;
}
