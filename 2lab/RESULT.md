# Результаты выполнения 2 лабы

## k-NN

```bash
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/k-NN$ g++ -fopenmp -o knn main.cpp knn.cpp
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/k-NN$ export OMP_NUM_THREADS=4
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/k-NN$ ./knn 
Предсказываемый класс: 0
```

## spam

```bash
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/spam$ g++ -fopenmp -o spam main.cpp spam_filter.cpp
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/spam$ export OMP_NUM_THREADS=4
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/spam$ ./spam
Email: Get cheap viagra now!!! Special offer!!!
Classification: HAM
```

## sast

```bash
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/spam$ g++ -fopenmp -o spam main.cpp spam_filter.cpp
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/spam$ export OMP_NUM_THREADS=4
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/spam$ ./spam
Email: Get cheap viagra now!!! Special offer!!!
Classification: HAM
```

## sast

```bash
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/sast$ ./sast wireshark-wireshark-4.4.9/
Сканирование директории: wireshark-wireshark-4.4.9/
Найдено файлов для анализа: 3939
Анализ завершён. Найдено проблем: 50360

=== РЕЗУЛЬТАТЫ SAST АНАЛИЗА ===

--- COMMAND_INJECTION (8527 найденных) ---
wireshark-wireshark-4.4.9/wiretap/merge.h:84: Потенциальная командная инъекция: exec
wireshark-wireshark-4.4.9/wiretap/merge.h:112: Потенциальная командная инъекция: exec
wireshark-wireshark-4.4.9/wiretap/merge.h:136: Потенциальная командная инъекция: exec
wireshark-wireshark-4.4.9/wiretap/merge.h:155: Потенциальная командная инъекция: exec
wireshark-wireshark-4.4.9/wiretap/wtap_opttypes.c:2166: Потенциальная командная инъекция: system
wireshark-wireshark-4.4.9/wiretap/nettl.h:18: Потенциальная командная инъекция: system
wireshark-wireshark-4.4.9/wiretap/blf.c:143: Потенциальная командная инъекция: system
wireshark-wireshark-4.4.9/wsutil/ws_assert.h:33: Потенциальная командная инъекция: exec
wireshark-wireshark-4.4.9/wsutil/ws_assert.h:37: Потенциальная командная инъекция: exec
wireshark-wireshark-4.4.9/wsutil/ws_assert.h:80: Потенциальная командная инъекция: exec
   ... и ещё 8517

--- FORMAT_STRING (191 найденных) ---
wireshark-wireshark-4.4.9/wiretap/daintree-sna.c:176: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/wiretap/netscreen.c:306: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/wiretap/cllog.c:258: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/wiretap/cllog.c:627: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/wiretap/cllog.c:744: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/wiretap/cllog.c:747: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/wiretap/cllog.c:750: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/capture/ws80211_utils.c:724: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/ui/cli/tap-iostat.c:1399: Небезопасный ввод: scanf
wireshark-wireshark-4.4.9/ui/decode_as_utils.c:308: Небезопасный ввод: scanf
   ... и ещё 181

--- BUFFER_OVERFLOW (1249 найденных) ---
wireshark-wireshark-4.4.9/wiretap/daintree-sna.c:79: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/daintree-sna.c:91: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/daintree-sna.c:166: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/erf.c:3172: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/netscreen.c:107: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/netscreen.c:136: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/netscreen.c:226: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/netscreen.c:341: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/netscreen.c:362: Небезопасная функция: gets
wireshark-wireshark-4.4.9/wiretap/cllog.c:82: Небезопасная функция: gets
   ... и ещё 1239

--- MEMORY_LEAK (27878 найденных) ---
wireshark-wireshark-4.4.9/file.h:251: new без проверки delete в функции
wireshark-wireshark-4.4.9/file.h:254: new без проверки delete в функции
wireshark-wireshark-4.4.9/file.h:275: new без проверки delete в функции
wireshark-wireshark-4.4.9/file.h:277: new без проверки delete в функции
wireshark-wireshark-4.4.9/file.h:453: new без проверки delete в функции
wireshark-wireshark-4.4.9/file.h:676: malloc без проверки free в функции
wireshark-wireshark-4.4.9/file.h:723: new без проверки delete в функции
wireshark-wireshark-4.4.9/file.h:729: new без проверки delete в функции
wireshark-wireshark-4.4.9/wiretap/autosar_dlt.c:182: malloc без проверки free в функции
wireshark-wireshark-4.4.9/wiretap/autosar_dlt.c:193: malloc без проверки free в функции
   ... и ещё 27868

---  (12515 найденных) ---
wireshark-wireshark-4.4.9/extcap_parser.h:163: 
wireshark-wireshark-4.4.9/extcap_parser.h:170: 
wireshark-wireshark-4.4.9/extcap_parser.h:186: 
wireshark-wireshark-4.4.9/extcap_parser.h:189: 
wireshark-wireshark-4.4.9/extcap_parser.h:192: 
wireshark-wireshark-4.4.9/file.h:298: 
wireshark-wireshark-4.4.9/file.h:309: 
wireshark-wireshark-4.4.9/file.h:324: 
wireshark-wireshark-4.4.9/file.h:703: 
wireshark-wireshark-4.4.9/file.h:714: 
   ... и ещё 12505

=== СВОДНАЯ СТАТИСТИКА ===
Всего проблем безопасности: 50360
COMMAND_INJECTION: 8527
FORMAT_STRING: 191
BUFFER_OVERFLOW: 1249
MEMORY_LEAK: 27878
: 12515
```

## anti_fraud

```bash
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/2lab/anti_fraud$ ./anti_fraud 
Сгенерировано 10000 тестовых транзакций
=== ЗАПУСК АНТИФРОД СИСТЕМЫ ===
Обработка 10000 транзакций в 10 пакетах
Анализ завершён за 20 мс
Использовано потоков: 4

=== РЕЗУЛЬТАТЫ АНТИФРОД АНАЛИЗА ===
Всего проанализировано транзакций: 10000
Подозрительных (риск > 0.5): 5518
Заблокировано (риск > 0.8): 1777
Средний риск: 0.567865

--- ТОП-10 САМЫХ РИСКОВАННЫХ ТРАНЗАКЦИЙ ---
Транзакция 7758 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 5.520625 раз, Транзакция из подозрительной страны: UA, Подозрительный мерчант: digital_services
Транзакция 4278 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 22.053475 раз, Транзакция из подозрительной страны: NG, Подозрительный мерчант: cryptocurrency
Транзакция 750 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 6.193399 раз, Транзакция из подозрительной страны: CN, Подозрительный мерчант: cryptocurrency
Транзакция 5214 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 30.598928 раз, Транзакция из подозрительной страны: CN, Подозрительный мерчант: digital_services
Транзакция 128 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 10.962309 раз, Транзакция из подозрительной страны: UA, Подозрительный мерчант: online_gambling
Транзакция 7709 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 5.126604 раз, Транзакция из подозрительной страны: TR, Подозрительный мерчант: digital_services
Транзакция 6317 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 21.852183 раз, Транзакция из подозрительной страны: CN, Подозрительный мерчант: cryptocurrency
Транзакция 8815 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 6.248142 раз, Транзакция из подозрительной страны: NG, Подозрительный мерчант: cryptocurrency
Транзакция 2985 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 98.893316 раз, Транзакция из подозрительной страны: NG, Подозрительный мерчант: digital_services
Транзакция 55 - Риск: 1.2 - Правил: 6 [ЗАБЛОКИРОВАНА]
   Факторы: Сумма превышает среднюю в 24.345032 раз, Транзакция из подозрительной страны: TR, Подозрительный мерчант: online_gambling
Результаты экспортированы в fraud_results.csv
```
