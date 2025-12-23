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
#include <future>
#include <atomic>
#include <memory>
#include <chrono>
#include <sstream>

// Константы для ограничений
constexpr int MAX_PAGES = 50;
constexpr size_t MAX_QUEUE_SIZE = 1000;  // Ограничение размера очереди (back-pressure)
constexpr long REQUEST_TIMEOUT = 10L;

// Структура для хранения информации об ошибках
struct RequestResult {
    std::string data;
    bool success;
    std::string error_message;
    long http_code;
};

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t newLength = size * nmemb;
    try {
        data->append(static_cast<char*>(contents), newLength);
        return newLength;
    } catch (const std::bad_alloc& e) {
        return 0;
    }
}

// RAII обертка для CURL
class CurlHandle {
public:
    CurlHandle() : curl(curl_easy_init()) {}
    ~CurlHandle() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }

    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;

    CURL* get() { return curl; }
    operator bool() const { return curl != nullptr; }

private:
    CURL* curl;
};

// Улучшенная функция запроса с детальным логированием ошибок
RequestResult make_get(const std::string& url) {
    RequestResult result;
    result.success = false;
    result.http_code = 0;

    CurlHandle curl;

    if (!curl) {
        result.error_message = "Failed to initialize CURL handle";
        std::cerr << "[ERROR] " << url << ": " << result.error_message << std::endl;
        return result;
    }

    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &result.data);
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "libcurl-link-parser/1.0");
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, REQUEST_TIMEOUT);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 5L);
    // Игнорируем SSL ошибки для простоты (в продакшене нужно настроить сертификаты)
    curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 0L);

    CURLcode res = curl_easy_perform(curl.get());

    if (res != CURLE_OK) {
        result.error_message = curl_easy_strerror(res);
        std::cerr << "[ERROR] " << url << ": " << result.error_message
                  << " (code: " << res << ")" << std::endl;
        return result;
    }

    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &result.http_code);

    if (result.http_code >= 400) {
        result.error_message = "HTTP error: " + std::to_string(result.http_code);
        std::cerr << "[WARN] " << url << ": " << result.error_message << std::endl;
        return result;
    }

    result.success = true;
    return result;
}

std::string getDomain(const std::string& url) {
    size_t start = url.find("://");
    if (start == std::string::npos) {
        start = 0;
    } else {
        start += 3;
    }
    size_t end = url.find('/', start);
    if (end == std::string::npos) {
        return url.substr(start);
    }
    return url.substr(start, end - start);
}

// Разрешает .. и . в пути URL
std::string resolvePath(const std::string& path) {
    std::vector<std::string> segments;
    std::string segment;
    std::istringstream stream(path);

    while (std::getline(stream, segment, '/')) {
        if (segment == "..") {
            if (!segments.empty() && segments.back() != "..") {
                segments.pop_back();
            }
        } else if (segment != "." && !segment.empty()) {
            segments.push_back(segment);
        }
    }

    std::string result;
    for (const auto& seg : segments) {
        result += "/" + seg;
    }

    return result.empty() ? "/" : result;
}

std::string normalizeUrl(const std::string& baseUrl, const std::string& link) {
    // Пропускаем якоря и javascript
    if (link.empty() || link[0] == '#' || link.find("javascript:") == 0) {
        return "";
    }

    std::string result;

    if (link.find("http://") == 0 || link.find("https://") == 0) {
        result = link;
    } else if (link.find("//") == 0) {
        size_t protocol_end = baseUrl.find("://");
        if (protocol_end == std::string::npos) {
            result = "https:" + link;
        } else {
            std::string protocol = baseUrl.substr(0, protocol_end);
            result = protocol + ":" + link;
        }
    } else if (!link.empty() && link[0] == '/') {
        size_t protocol_end = baseUrl.find("://");
        if (protocol_end == std::string::npos) {
            result = link;
        } else {
            size_t domain_end = baseUrl.find('/', protocol_end + 3);
            if (domain_end == std::string::npos) {
                result = baseUrl + link;
            } else {
                result = baseUrl.substr(0, domain_end) + link;
            }
        }
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
        result = base + link;
    }

    // Удаляем якорь из URL
    size_t hash_pos = result.find('#');
    if (hash_pos != std::string::npos) {
        result = result.substr(0, hash_pos);
    }

    // Разрешаем .. и . в пути
    size_t protocol_end = result.find("://");
    if (protocol_end != std::string::npos) {
        size_t path_start = result.find('/', protocol_end + 3);
        if (path_start != std::string::npos) {
            std::string base_part = result.substr(0, path_start);
            std::string path_part = result.substr(path_start);

            // Отделяем query string
            std::string query;
            size_t query_pos = path_part.find('?');
            if (query_pos != std::string::npos) {
                query = path_part.substr(query_pos);
                path_part = path_part.substr(0, query_pos);
            }

            path_part = resolvePath(path_part);
            result = base_part + path_part + query;
        }
    }

    return result;
}

// Проверка, является ли путь "мусорным" (HTML теги, JS код и т.д.)
bool isJunkPath(const std::string& path) {
    // Список HTML тегов и прочего мусора
    static const std::unordered_set<std::string> junk = {
        // HTML теги
        "a", "b", "i", "p", "q", "s", "u", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
        "em", "li", "ol", "ul", "dd", "dl", "dt", "td", "th", "tr", "tt",
        "title", "script", "style", "head", "body", "html", "div", "span",
        "svg", "path", "g", "figure", "img", "link", "meta", "noscript",
        "form", "input", "button", "select", "option", "textarea", "label",
        "table", "thead", "tbody", "tfoot", "caption", "col", "colgroup",
        "nav", "header", "footer", "main", "article", "section", "aside",
        "pre", "code", "kbd", "samp", "var", "cite", "abbr", "dfn",
        "iframe", "embed", "object", "param", "video", "audio", "source", "track",
        "canvas", "map", "area", "picture", "figcaption", "details", "summary",
        "dialog", "menu", "menuitem", "slot", "template", "blockquote",
        // Расширения файлов
        "png", "jpg", "jpeg", "gif", "css", "js", "ico", "woff", "woff2",
        "ttf", "eot", "mp4", "webm", "mp3", "ogg", "pdf", "xml", "json"
    };

    // Извлекаем последний сегмент пути
    std::string segment = path;
    size_t lastSlash = path.find_last_of('/');
    if (lastSlash != std::string::npos && lastSlash + 1 < path.size()) {
        segment = path.substr(lastSlash + 1);
    }

    // Убираем query string
    size_t queryPos = segment.find('?');
    if (queryPos != std::string::npos) {
        segment = segment.substr(0, queryPos);
    }

    // Приводим к нижнему регистру для сравнения
    std::string lower_segment;
    lower_segment.reserve(segment.size());
    for (char c : segment) {
        lower_segment += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // Проверяем на мусор (HTML теги)
    if (junk.count(lower_segment) > 0) return true;

    // Слишком короткий путь (1-2 символа) - скорее всего мусор
    if (segment.length() <= 2) return true;

    // Проверяем на подозрительные паттерны (JS код, base64 и т.д.)
    if (segment.find("function") != std::string::npos) return true;
    if (segment.find("var ") != std::string::npos) return true;
    if (segment.find("==") != std::string::npos) return true;
    if (segment.find(";") != std::string::npos) return true;
    if (segment.find("{") != std::string::npos) return true;
    if (segment.find("}") != std::string::npos) return true;
    if (segment.find("(") != std::string::npos) return true;
    if (segment.find(")") != std::string::npos) return true;
    if (segment.find("+") != std::string::npos && segment.length() > 50) return true;
    if (segment.length() > 100) return true;  // Слишком длинный сегмент - скорее всего мусор

    return false;
}

// Извлечение URL с возвратом результата без исключений
std::vector<std::string> extractURLs(const std::string& url, const std::string& target_domain,
                                      bool& success, std::string& error_msg) {
    std::vector<std::string> links;
    success = false;

    RequestResult response = make_get(url);

    if (!response.success) {
        error_msg = response.error_message;
        return links;
    }

    if (response.data.empty()) {
        error_msg = "Empty response";
        return links;
    }

    try {
        std::unordered_set<std::string> unique_links;

        // Регекс 1: ссылки в href атрибутах (основной способ)
        std::regex hrefPattern(R"(href\s*=\s*["']([^"']+)["'])", std::regex_constants::icase);

        auto href_begin = std::sregex_iterator(response.data.begin(), response.data.end(), hrefPattern);
        auto href_end = std::sregex_iterator();

        for (std::sregex_iterator i = href_begin; i != href_end; ++i) {
            std::smatch match = *i;
            if (match.size() > 1) {
                std::string raw_link = match[1].str();
                std::string absolute_url = normalizeUrl(url, raw_link);

                if (absolute_url.empty()) continue;
                if (isJunkPath(absolute_url)) continue;

                std::string link_domain = getDomain(absolute_url);
                if (link_domain == target_domain) {
                    if (unique_links.find(absolute_url) == unique_links.end()) {
                        unique_links.insert(absolute_url);
                        links.push_back(absolute_url);
                    }
                }
            }
        }

        success = true;
    } catch (const std::regex_error& e) {
        error_msg = std::string("Regex error: ") + e.what();
        std::cerr << "[ERROR] " << error_msg << std::endl;
    } catch (const std::exception& e) {
        error_msg = std::string("Exception: ") + e.what();
        std::cerr << "[ERROR] " << error_msg << std::endl;
    }

    return links;
}

// Структура для древовидного представления сайта
struct SiteNode {
    std::string url;
    std::vector<std::string> children;  // Ссылки, найденные на этой странице
};

// Сохранение результатов в XML
void saveToXML(const std::string& filename,
               const std::string& domain,
               const std::unordered_map<std::string, std::vector<std::string>>& site_map) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<sitemap domain=\"" << domain << "\">\n";

    for (const auto& entry : site_map) {
        file << "  <page url=\"" << entry.first << "\">\n";
        file << "    <links count=\"" << entry.second.size() << "\">\n";
        for (const auto& link : entry.second) {
            file << "      <link>" << link << "</link>\n";
        }
        file << "    </links>\n";
        file << "  </page>\n";
    }

    file << "</sitemap>\n";
    file.close();

    std::cout << "Sitemap saved to: " << filename << std::endl;
}

int main() {
    // Инициализация CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::string start_url;

    std::cout << "Введите URL: ";
    std::cin >> start_url;

    std::string target_domain = getDomain(start_url);
    std::cout << "Целевой домен: " << target_domain << std::endl;

    // Структуры данных
    std::unordered_set<std::string> visited;
    std::queue<std::string> to_visit;
    std::vector<std::string> all_links;
    std::unordered_map<std::string, std::vector<std::string>> site_map;  // Для древовидной структуры

    std::mutex mtx;
    std::mutex cout_mtx;  // Мьютекс для вывода в консоль
    std::condition_variable cv;
    std::condition_variable cv_queue_space;  // Для back-pressure

    std::atomic<int> processed_pages{0};  // Успешно обработанные
    std::atomic<int> task_number{0};      // Номер текущей задачи для вывода
    std::atomic<int> active_tasks{0};
    std::atomic<bool> done{false};

    to_visit.push(start_url);
    visited.insert(start_url);
    all_links.push_back(start_url);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (unsigned int t = 0; t < numThreads; t++) {
        threads.emplace_back([&]() {
            while (true) {
                std::string current_url;
                bool should_notify = false;

                {
                    std::unique_lock<std::mutex> lock(mtx);

                    // Ждём пока есть работа или done
                    cv.wait(lock, [&]() {
                        return !to_visit.empty() || done.load() ||
                               (to_visit.empty() && active_tasks.load() == 0);
                    });

                    // Условие выхода: done И очередь пуста И нет активных задач
                    if (to_visit.empty()) {
                        if (active_tasks.load() == 0) {
                            done.store(true);
                            should_notify = true;
                        }
                        if (done.load()) {
                            // Выходим только если done и нет работы
                            lock.unlock();
                            if (should_notify) cv.notify_all();
                            return;
                        }
                        // Иначе продолжаем ждать
                        continue;
                    }

                    // Проверяем лимит до взятия URL
                    if (processed_pages.load() >= MAX_PAGES) {
                        done.store(true);
                        should_notify = true;
                        lock.unlock();
                        if (should_notify) cv.notify_all();
                        return;
                    }

                    current_url = to_visit.front();
                    to_visit.pop();
                    active_tasks.fetch_add(1);

                    // Сигнализируем что есть место в очереди (для back-pressure)
                    if (to_visit.size() < MAX_QUEUE_SIZE) {
                        cv_queue_space.notify_one();
                    }
                }

                // Получаем уникальный номер задачи атомарно
                int my_task_num = task_number.fetch_add(1) + 1;

                // Обработка страницы вне мьютекса
                {
                    std::lock_guard<std::mutex> cout_lock(cout_mtx);
                    std::cout << "[" << my_task_num << "/" << MAX_PAGES
                              << "] Обработка: " << current_url << std::endl;
                }

                std::vector<std::string> new_links;
                bool extraction_success = false;
                std::string error_msg;

                // try-catch для защиты от исключений
                try {
                    new_links = extractURLs(current_url, target_domain, extraction_success, error_msg);
                } catch (const std::exception& e) {
                    std::cerr << "[EXCEPTION] " << current_url << ": " << e.what() << std::endl;
                    extraction_success = false;
                } catch (...) {
                    std::cerr << "[EXCEPTION] " << current_url << ": Unknown error" << std::endl;
                    extraction_success = false;
                }

                // Увеличиваем счётчик ПОСЛЕ успешной обработки
                if (extraction_success) {
                    processed_pages.fetch_add(1);
                }

                {
                    std::unique_lock<std::mutex> lock(mtx);

                    // Сохраняем в карту сайта
                    site_map[current_url] = new_links;

                    for (const std::string& link : new_links) {
                        // Back-pressure: ждём если очередь слишком большая
                        while (to_visit.size() >= MAX_QUEUE_SIZE && !done.load()) {
                            cv_queue_space.wait_for(lock, std::chrono::milliseconds(100));
                        }

                        if (done.load()) break;

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
                        should_notify = true;
                    }
                }

                // notify_all() ПОСЛЕ освобождения мьютекса для уменьшения contention
                cv.notify_all();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Вывод результатов
    std::cout << "\n========================================" << std::endl;
    std::cout << "Найдено " << all_links.size() << " уникальных ссылок для домена "
              << target_domain << std::endl;
    std::cout << "Обработано страниц: " << processed_pages.load() << std::endl;
    std::cout << "========================================\n" << std::endl;

    for (const std::string& link : all_links) {
        std::cout << link << std::endl;
    }

    // Сохранение в XML
    saveToXML("sitemap.xml", target_domain, site_map);

    // Очистка CURL
    curl_global_cleanup();

    return 0;
}
