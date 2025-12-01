#include <iostream>
#include <vector>
#include <cstring>
#include <random>

void obfuscate_data(char* input_data, char* output_data, int* keys, 
                   int num_items, int item_size) {
    for (int item_id = 0; item_id < num_items; item_id++) {
        char* input = &input_data[item_id * item_size];
        char* output = &output_data[item_id * item_size];
        int key = keys[item_id];
        
        // Простая обфускация: XOR с ключом
        for (int i = 0; i < item_size; i++) {
            output[i] = input[i] ^ (key + i);
        }
    }
}

class DataObfuscator {
public:
    std::vector<std::string> obfuscate(const std::vector<std::string>& data, 
                                      const std::vector<int>& keys) {
        int num_items = data.size();
        int item_size = 256;
        
        char* h_input_data = new char[num_items * item_size];
        char* h_output_data = new char[num_items * item_size];
        int* h_keys = new int[num_items];
        
        memset(h_input_data, 0, num_items * item_size);
        memset(h_output_data, 0, num_items * item_size);
        
        for (int i = 0; i < num_items; i++) {
            strncpy(&h_input_data[i * item_size], 
                   data[i].c_str(), 
                   data[i].size());
            h_input_data[i * item_size + data[i].size()] = '\0';
        }
        memcpy(h_keys, keys.data(), num_items * sizeof(int));
        
        obfuscate_data(h_input_data, h_output_data, h_keys, num_items, item_size);
        
        std::vector<std::string> result;
        for (int i = 0; i < num_items; i++) {
            result.push_back(std::string(&h_output_data[i * item_size]));
        }
        
        delete[] h_input_data;
        delete[] h_output_data;
        delete[] h_keys;
        
        return result;
    }
};
std::string generate_string(int length) {
    static const std::string characters = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        "!@#$%^&*()_+-=[]{}|;:,.<>?";
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, characters.size() - 1);
    
    std::string result;
    for (int i = 0; i < length; i++) {
        result += characters[dis(gen)];
    }
    return result;
}

int random_int(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}
int main() {
    DataObfuscator obfuscator;
    
    int num_items = 10;
    std::vector<std::string> original_data;
    std::vector<int> keys;
    
    std::cout << "Generated " << num_items << " items:" << std::endl;
    
    for (int i = 0; i < num_items; i++) {

        int str_length = random_int(5, 30);
        std::string random_str = generate_string(str_length);
        
        original_data.push_back(random_str);
        keys.push_back(10);
        
        std::cout << "  " << (i + 1) << ". \"" << original_data[i] 
                  << "\" (key: " << keys[i] << ", length: " << original_data[i].length() << ")" << std::endl;
    }
    
    std::vector<std::string> obfuscated = obfuscator.obfuscate(original_data, keys);
    
    std::cout << "\nОбфусцированные данные:" << std::endl;
    for (const auto& str : obfuscated) {
        std::cout << "  ";
        for (char c : str) {
            if (c >= 32 && c <= 126) {
                std::cout << c;
            } else {
                std::cout << "?";
            }
        }
        std::cout << " (hex: ";
        for (size_t i = 0; i < str.size() && i < 10; i++) {
            printf("%02x ", (unsigned char)str[i]);
        }
        std::cout << ")" << std::endl;
    }
    
    std::cout << "\nДемонстрация обратимости:" << std::endl;
    
    std::vector<std::string> deobfuscated = obfuscator.obfuscate(obfuscated, keys);
    
    std::cout << "Деобфусцированные данные:" << std::endl;
    for (const auto& str : deobfuscated) {
        std::cout << "  " << str << std::endl;
    }
    
    bool success = true;
    for (size_t i = 0; i < original_data.size(); i++) {
        if (original_data[i] != deobfuscated[i]) {
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "\n✓ Обфускация обратима - все данные восстановлены корректно!" << std::endl;
    } else {
        std::cout << "\n✗ Ошибка: восстановление данных не удалось!" << std::endl;
    }
    
    
    return 0;
}