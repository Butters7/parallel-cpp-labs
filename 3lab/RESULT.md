# Результаты 3 лабы

## nmap

```bash
mkdir -p build && cd build
cmake ..
make
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/3lab/nmap$ mpic++ -o nmap nmap.cpp
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/3lab/nmap$ mpirun -n 4 ./nmap
Worker 2: Сканирую 192.168.1.2
=== MASTER: Запуск распределённого сканирования ===
Количество worker'ов: 3
Всего IP для сканирования: 15
MASTER: Отправил задание 192.168.1.1 worker'у 1
MASTER: Отправил задание 192.168.1.2 worker'у 2
MASTER: Отправил задание 192.168.1.3 worker'у 3
Worker 1: Сканирую 192.168.1.1
Worker 3: Сканирую 192.168.1.3
Worker 2: Сканирую 192.168.1.4
MASTER: Получен результат от worker'а 2 для 192.168.1.2 (время: 3.55543с)
MASTER: Отправил новое задание 192.168.1.4 worker'у 2
Worker 1: Сканирую 192.168.1.5
MASTER: Получен результат от worker'а 1 для 192.168.1.1 (время: 3.72399с)
MASTER: Отправил новое задание 192.168.1.5 worker'у 1
MASTER: Получен результат от worker'а 3 для 192.168.1.3 (время: 3.72836с)
MASTER: Отправил новое задание 192.168.2.1 worker'у 3
Worker 3: Сканирую 192.168.2.1
MASTER: Получен результат от worker'а 3 для 192.168.2.1 (время: 3.91129с)
MASTER: Отправил новое задание 192.168.2.2 worker'у 3
Worker 3: Сканирую 192.168.2.2
MASTER: Получен результат от worker'а 2 для 192.168.1.4 (время: 4.11793с)
MASTER: Отправил новое задание 192.168.2.3 worker'у 2
Worker 2: Сканирую 192.168.2.3
Worker 1: Сканирую 192.168.2.4
MASTER: Получен результат от worker'а 1 для 192.168.1.5 (время: 3.94765с)
MASTER: Отправил новое задание 192.168.2.4 worker'у 1
MASTER: Получен результат от worker'а 1 для 192.168.2.4 (время: 2.58511с)
MASTER: Отправил новое задание 192.168.2.5 worker'у 1
Worker 1: Сканирую 192.168.2.5
MASTER: Получен результат от worker'а 2 для 192.168.2.3 (время: 2.59959с)
MASTER: Отправил новое задание 10.0.0.1 worker'у 2
Worker 2: Сканирую 10.0.0.1
Worker 3: Сканирую 10.0.0.2
MASTER: Получен результат от worker'а 3 для 192.168.2.2 (время: 2.64553с)
MASTER: Отправил новое задание 10.0.0.2 worker'у 3
MASTER: Получен результат от worker'а 1 для 192.168.2.5 (время: 3.17302с)
MASTER: Отправил новое задание 10.0.0.3 worker'у 1
Worker 1: Сканирую 10.0.0.3
MASTER: Получен результат от worker'а 3 для 10.0.0.2 (время: 3.17253с)
MASTER: Отправил новое задание 10.0.0.4 worker'у 3
Worker 3: Сканирую 10.0.0.4
MASTER: Получен результат от worker'а 3 для 10.0.0.4 (время: 4.49519с)
MASTER: Отправил новое задание 10.0.0.5 worker'у 3
Worker 3: Сканирую 10.0.0.5
MASTER: Получен результат от worker'а 1 для 10.0.0.3 (время: 4.53429с)
MASTER: Получен результат от worker'а 3 для 10.0.0.5 (время: 3.51116с)
Worker 3: Получен сигнал завершения
MASTER: Получен результат от worker'а 2 для 10.0.0.1 (время: 28.3392с)

=== РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ ===

[Worker 2] 192.168.1.2:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 2.54 seconds


[Worker 1] 192.168.1.1:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 2.71 seconds


[Worker 3] 192.168.1.3:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 2.71 seconds


[Worker 3] 192.168.2.1:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 3.70 seconds


[Worker 2] 192.168.1.4:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 3.91 seconds


[Worker 1] 192.168.1.5:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 3.71 seconds


[Worker 1] 192.168.2.4:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 2.44 seconds


[Worker 2] 192.168.2.3:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 2.44 seconds


[Worker 3] 192.168.2.2:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 2.46 seconds


[Worker 1] 192.168.2.5:
Worker 2: Получен сигнал завершения
Worker 1: Получен сигнал завершения
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 3.05 seconds


[Worker 3] 10.0.0.2:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 3.04 seconds


[Worker 3] 10.0.0.4:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:30 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 4.37 seconds


[Worker 1] 10.0.0.3:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:30 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 4.37 seconds


[Worker 3] 10.0.0.5:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:30 MSK
Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn
Nmap done: 1 IP address (0 hosts up) scanned in 2.46 seconds


[Worker 2] 10.0.0.1:
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-11-30 01:29 MSK
Nmap scan report for 10.0.0.1
Host is up (0.062s latency).
Not shown: 989 filtered tcp ports (no-response)
PORT     STATE  SERVICE    VERSION
53/tcp   closed domain
80/tcp   open   http       nginx
|_http-title: 403 Forbidden
5900/tcp open   vnc        VNC (protocol 3.8)
| vnc-info: 
|   Protocol version: 3.8
|   Security types: 
|_    VNC Authentication (2)
5901/tcp open   vnc        VNC (protocol 3.8)
| vnc-info: 
|   Protocol ...

=== СТАТИСТИКА ===
Успешных сканирований: 15 из 15

Задач на каждого worker'а:
  Worker 1: 5 задач
  Worker 2: 4 задач
  Worker 3: 6 задач
```

## ddos

```bash
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/3lab/ddos/build$ mpirun -n 4 ./ddos
=== MASTER: Запуск распределённого анализа DDoS ===
Количество worker'ов: 3
Всего пакетов для анализа: 10000
Генерация трафика завершена за 5 мс
Worker 2: Получил 3333 пакетов для анализа
Worker 2: Анализ завершён, обнаружено атак: 0
Worker 2: Получен сигнал завершения
Worker 3: Получил 3333 пакетов для анализа
Worker 3: Анализ завершён, обнаружено атак: 0
Worker 3: Получен сигнал завершения
MASTER: Отправил 3334 пакетов worker'у 1
MASTER: Отправил 3333 пакетов worker'у 2
MASTER: Отправил 3333 пакетов worker'у 3
MASTER: Получил статистику от worker'а 1
MASTER: Получил статистику от worker'а 2
MASTER: Получил статистику от worker'а 3

========================================
=== ГЛОБАЛЬНАЯ СТАТИСТИКА DDoS АНАЛИЗА ===
========================================

Время анализа: 5 мс
Обработано пакетов: 9517
Обнаружено DDoS атак (rate limit): 0
Обнаружено SYN-flood атак: 0

--- Статистика по worker'ам ---
Worker 1:
  Пакетов обработано: 3226
  Атак обнаружено: 0
  Топ-атакующий: 192.168.100.5 (14 пакетов)
Worker 2:
  Пакетов обработано: 3151
  Атак обнаружено: 0
  Топ-атакующий: 10.0.86.33 (3 пакетов)
Worker 3:
  Пакетов обработано: 3140
  Атак обнаружено: 0
  Топ-атакующий: 192.168.100.2 (3 пакетов)

DDoS атаки не обнаружены
Worker 1: Получил 3334 пакетов для анализа
Worker 1: Анализ завершён, обнаружено атак: 0
Worker 1: Получен сигнал завершения
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/3lab/ddos/build$ mpirun -n 4 ./ddos 50000
=== MASTER: Запуск распределённого анализа DDoS ===
Количество worker'ов: 3
Всего пакетов для анализа: 50000
Генерация трафика завершена за 19 мс
Worker 1: Получил 16667 пакетов для анализа
MASTER: Отправил 16667 пакетов worker'у 1
Worker 2: Получил 16667 пакетов для анализа
[Worker 2] ALERT: Rate limit exceeded for IP: 192.168.100.1
Worker 3: Получил 16666 пакетов для анализа
MASTER: Отправил 16667 пакетов worker'у 2
MASTER: Отправил 16666 пакетов worker'у 3
[Worker 1] ALERT: Rate limit exceeded for IP: 192.168.100.1
[Worker 1] ALERT: Rate limit exceeded for IP: 192.168.100.2
[Worker 1] ALERT: Rate limit exceeded for IP: 192.168.100.3
[Worker 1] ALERT: Rate limit exceeded for IP: 192.168.100.4
[Worker 1] ALERT: Rate limit exceeded for IP: 192.168.100.5
[Worker 2] ALERT: Rate limit exceeded for IP: 192.168.100.2
[Worker 2] ALERT: Rate limit exceeded for IP: 192.168.100.3
[Worker 2] ALERT: Rate limit exceeded for IP: 192.168.100.4
[Worker 2] ALERT: Rate limit exceeded for IP: 192.168.100.5
[Worker 3] ALERT: Rate limit exceeded for IP: 192.168.100.1
[Worker 3] ALERT: Rate limit exceeded for IP: 192.168.100.2
[Worker 3] ALERT: Rate limit exceeded for IP: 192.168.100.3
[Worker 3] ALERT: Rate limit exceeded for IP: 192.168.100.4
[Worker 3] ALERT: Rate limit exceeded for IP: 192.168.100.5
Worker 1: Анализ завершён, обнаружено атак: 5
MASTER: Получил статистику от worker'а 1
Worker 3: Анализ завершён, обнаружено атак: 5
MASTER: Получил статистику от worker'а 2
Worker 2: Анализ завершён, обнаружено атак: 5
MASTER: Получил статистику от worker'а 3

========================================
=== ГЛОБАЛЬНАЯ СТАТИСТИКА DDoS АНАЛИЗА ===
========================================

Время анализа: 30 мс
Обработано пакетов: 48129
Worker 1: Получен сигнал завершения
Worker 2: Получен сигнал завершения
Worker 3: Получен сигнал завершения
Обнаружено DDoS атак (rate limit): 15
Обнаружено SYN-flood атак: 0

--- Статистика по worker'ам ---
Worker 1:
  Пакетов обработано: 16095
  Атак обнаружено: 5
  Топ-атакующий: 192.168.100.3 (147 пакетов)
Worker 2:
  Пакетов обработано: 16048
  Атак обнаружено: 5
  Топ-атакующий: 192.168.100.5 (147 пакетов)
Worker 3:
  Пакетов обработано: 15986
  Атак обнаружено: 5
  Топ-атакующий: 192.168.100.2 (136 пакетов)

!!! DDoS АТАКА ОБНАРУЖЕНА !!!
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/3lab/ddos/build$ 
```

## sqlmap

```bash
mkdir -p build && cd build
cmake ..
make
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/3lab/sqlmap$ mpirun -n 4 ./build/sqli 
========================================
  Distributed SQLMap Scanner (MPI)
========================================

Проверка доступности sqlmap...
sqlmap доступен
=== MASTER: Запуск распределённого SQL-инъекция сканирования ===
Количество worker'ов: 3
Всего целей для сканирования: 120
Профиль: Quick (level 1-2)
MASTER: Отправил задание 0 worker'у 1
MASTER: Отправил задание 1 worker'у 2
MASTER: Отправил задание 2 worker'у 3
[Worker 2] Scanning: http://127.0.0.1:4280/vulnerabilities/brute/ (Profile: Quick (level 1-2))
[Worker 3] Scanning: http://127.0.0.1:4280/vulnerabilities/exec/ (Profile: Quick (level 1-2))
[Worker 1] Scanning: http://127.0.0.1:4280/setup.php/ (Profile: Quick (level 1-2))

--- Реал-тайм статистика ---
Прогресс: 1/120 (0%)
Время: 0.493774 сек
Worker 1: завершил задание 0 за 0.493322 сек
MASTER: Отправил новое задание 3 worker'у 1
[Worker 1] Scanning: http://127.0.0.1:4280/vulnerabilities/csrf/ (Profile: Quick (level 1-2))

--- Реал-тайм статистика ---
Прогресс: 2/120 (1%)
Время: 0.571788 сек
Worker 3: завершил задание 2 за 0.570858 сек
[Worker 3] Scanning: http://127.0.0.1:4280/vulnerabilities/upload/ (Profile: Quick (level 1-2))
MASTER: Отправил новое задание 4 worker'у 3

--- Реал-тайм статистика ---
Прогресс: 3/120 (2%)
Время: 0.62 сек
Worker 2: завершил задание 1 за 0.619266 сек
MASTER: Отправил новое задание 5 worker'у 2
[Worker 2] Scanning: http://127.0.0.1:4280/vulnerabilities/captcha/ (Profile: Quick (level 1-2))
[Worker 3] Scanning: http://127.0.0.1:4280/vulnerabilities/sqli/ (Profile: Quick (level 1-2))
...
--- Реал-тайм статистика ---
Прогресс: 119/120 (99%)
Время: 21.6371 сек
Worker 2: завершил задание 118 за 0.706236 сек

--- Реал-тайм статистика ---
Прогресс: 120/120 (100%)
Время: 21.6432 сек
Worker 3: завершил задание 119 за 0.564081 сек

========================================
=== ИТОГОВАЯ СТАТИСТИКА ===
========================================
Общее время: 21.6439 сек
Просканировано целей: 120
Найдено уязвимостей: 0
Средняя скорость: 5.54429 целей/сек

--- Загрузка процессов ---
Worker 1: 40 задач, 21.4221 сек
Worker 2: 40 задач, 21.3981 сек
Worker 3: 40 задач, 21.465 сек

Отчёт сохранён в: sqlmap_mpi_report.txt
```

## ioc

```bash
bttrs@bttrs:~/Desktop/labs-plus-plus/labs/3lab/ioc/build$ mpirun -n 4 ./ioc /var/log iocs.csv
[2025-11-30 01:48:35] [INFO ] [Rank 0] === MASTER: Starting distributed IOC scan ===
[2025-11-30 01:48:35] [INFO ] [Rank 0] Workers: 3
[2025-11-30 01:48:35] [INFO ] [Rank 0] Loaded 10 default IOCs
[2025-11-30 01:48:35] [INFO ] [Rank 1] Worker started
[2025-11-30 01:48:35] [INFO ] [Rank 2] Worker started
[2025-11-30 01:48:35] [INFO ] [Rank 3] Worker started
[2025-11-30 01:48:35] [INFO ] [Rank 0] Files to scan: 271
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/boot.log.3
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/btmp.1
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 10/271 (3%)
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/boot.log.2
[2025-11-30 01:48:35] [WARN ] [Rank 2] Skipping symlink: /var/log/README
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 20/271 (7%)
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 30/271 (11%)
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 40/271 (14%)
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/boot.log.6
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/installer/subiquity-server-info.log.5759
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 50/271 (18%)
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/installer-journal.txt
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/installer/subiquity-server-debug.log.5759
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/subiquity-server-debug.log.4582
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/subiquity-server-info.log.4582
[2025-11-30 01:48:35] [WARN ] [Rank 2] Skipping symlink: /var/log/installer/ubuntu_bootstrap.log
[2025-11-30 01:48:35] [WARN ] [Rank 2] Skipping symlink: /var/log/installer/subiquity-server-info.log
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 60/271 (22%)
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/cloud-init.log
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/installer/autoinstall-user-data
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/cloud-init-output.log
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/curtin-install/subiquity-initial.conf
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/curtin-install/subiquity-extract.conf
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 70/271 (25%)
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/curtin-install/subiquity-curtin-apt.conf
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/curtin-install/subiquity-curthooks.conf
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/installer/curtin-install/subiquity-partitioning.conf
[2025-11-30 01:48:35] [WARN ] [Rank 2] Skipping symlink: /var/log/installer/subiquity-server-debug.log
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/boot.log
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 80/271 (29%)
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/boot.log.4
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 90/271 (33%)
[2025-11-30 01:48:35] [ERROR] [Rank 3] Cannot open file: /var/log/vboxadd-install.log
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/boot.log.7
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 100/271 (36%)
[2025-11-30 01:48:35] [ERROR] [Rank 2] Cannot open file: /var/log/boot.log.5
[2025-11-30 01:48:35] [INFO ] [Rank 0] Progress: 110/271 (40%)
[2025-11-30 01:48:37] [INFO ] [Rank 0] Progress: 120/271 (44%)
[2025-11-30 01:48:38] [INFO ] [Rank 0] Progress: 130/271 (47%)
[2025-11-30 01:48:39] [INFO ] [Rank 0] Progress: 140/271 (51%)
[2025-11-30 01:48:40] [INFO ] [Rank 0] Progress: 150/271 (55%)
[2025-11-30 01:48:41] [INFO ] [Rank 0] Progress: 160/271 (59%)
[2025-11-30 01:48:42] [INFO ] [Rank 0] Progress: 170/271 (62%)
[2025-11-30 01:48:43] [INFO ] [Rank 0] Progress: 180/271 (66%)
[2025-11-30 01:48:44] [INFO ] [Rank 0] Progress: 190/271 (70%)
[2025-11-30 01:48:46] [INFO ] [Rank 0] Progress: 200/271 (73%)
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 210/271 (77%)
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 220/271 (81%)
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 230/271 (84%)
[2025-11-30 01:48:47] [ERROR] [Rank 1] Cannot open file: /var/log/boot.log.1
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 240/271 (88%)
[2025-11-30 01:48:47] [ERROR] [Rank 1] Cannot open file: /var/log/btmp
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 250/271 (92%)
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 260/271 (95%)
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 270/271 (99%)
[2025-11-30 01:48:47] [INFO ] [Rank 0] Progress: 271/271 (100%)
[2025-11-30 01:48:47] [INFO ] [Rank 2] Received termination signal
[2025-11-30 01:48:47] [INFO ] [Rank 3] Received termination signal
[2025-11-30 01:48:47] [INFO ] [Rank 3] Worker finished
[2025-11-30 01:48:47] [INFO ] [Rank 1] Received termination signal
[2025-11-30 01:48:47] [INFO ] [Rank 1] Worker finished

========================================
=== IOC SCAN REPORT ===
========================================

Total time: 12.2358 sec
Files scanned: 271
IOC matches found: 0

--- Severity Statistics ---

--- Worker Load (balancing) ---
  Worker 1: 63 tasks, 11.6201 sec
  Worker 2: 104 tasks, 12.0794 sec
  Worker 3: 104 tasks, 11.6117 sec

--- Performance ---
Workers: 3
Avg time per worker: 4.07859 sec
[2025-11-30 01:48:47] [INFO ] [Rank 0] Report saved to ioc_report.txt
[2025-11-30 01:48:47] [INFO ] [Rank 2] Worker finished


```
