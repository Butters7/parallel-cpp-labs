#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <curl/curl.h>
#include <set>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <memory>

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t newLength = size * nmemb;
    try {
        data->append((char*)contents, newLength);
        return newLength;
    } catch (std::bad_alloc& e) {
        return 0;
    }
}

// RAII обертка для CURL - автоматически освобождает ресурсы
class CurlHandle {
public:
    CurlHandle() : curl(curl_easy_init()) {}
    ~CurlHandle() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }

    // Запрет копирования
    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;

    CURL* get() { return curl; }
    operator bool() const { return curl != nullptr; }

private:
    CURL* curl;
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

        CURLcode res = curl_easy_perform(curl.get());

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
    }
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
    // Инициализация CURL один раз для всей программы
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::string start_url;

    std::cout << "Ведите URL: ";
    std::cin >> start_url;

    std::string target_domain = getDomain(start_url);
    std::cout << "Текущий домен domain: " << target_domain << std::endl;

    std::set<std::string> visited;
    std::queue<std::string> to_visit;
    std::vector<std::string> all_links;

    std::mutex mtx;
    std::condition_variable cv;

    // Используем atomic для безопасного доступа из разных потоков
    std::atomic<int> processed_pages{0};
    std::atomic<int> active_tasks{0};
    std::atomic<bool> done{false};

    to_visit.push(start_url);
    visited.insert(start_url);
    all_links.push_back(start_url);

    const int max_pages = 50;

    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&]() {
            while (true) {
                std::string current_url;

                {
                    std::unique_lock<std::mutex> lock(mtx);

                    cv.wait(lock, [&]() {
                        return !to_visit.empty() || done.load();
                    });

                    if (done.load() && to_visit.empty()) {
                        return;
                    }

                    // Проверяем лимит до взятия URL
                    if (processed_pages.load() >= max_pages) {
                        done.store(true);
                        cv.notify_all();
                        return;
                    }

                    current_url = to_visit.front();
                    to_visit.pop();
                    processed_pages.fetch_add(1);
                    active_tasks.fetch_add(1);
                }

                std::cout << "Обработка: " << current_url << std::endl;

                // Скачиваем и парсим страницу (вне мьютекса)
                std::vector<std::string> new_links = extractURLs(current_url, target_domain);

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    for (const std::string& link : new_links) {
                        if (visited.find(link) == visited.end()) {
                            visited.insert(link);
                            to_visit.push(link);
                            all_links.push_back(link);
                        }
                    }

                    active_tasks.fetch_sub(1);

                    // Проверяем условие завершения
                    if (to_visit.empty() && active_tasks.load() == 0) {
                        done.store(true);
                    }
                }

                cv.notify_all();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\nНайдено " << all_links.size() << " уникальных ссылок для текущего домена " << target_domain << ":" << std::endl;
    for (const std::string& link : all_links) {
        std::cout << link << std::endl;
    }

    // Очистка CURL
    curl_global_cleanup();

    return 0;
}