# Primerjava zmogljivosti Bun.js in Node.js

Projekt vsebuje aplikaciji, merilne skripte in rezultate diplomskega dela. Primerjava vključuje čas zagona, latenco, prepustnost, procesorski čas in porabo pomnilnika.

## Struktura

```text
bun-node-performance-comparison-thesis/
├── README.md
├── docker-compose.yml
├── run-load-benchmarks.ps1
├── json-1mb.json
├── node-app/
├── bun-app/
├── data/
│   └── test-10mb.txt
├── startup-benchmark/
│   ├── benchmark.py
│   ├── app/server.mjs
│   └── results/
└── results/
    └── load-tests-20260904-052953/
```

## Obremenitveni testi

| Scenarij | Metoda | Povezave | Namen |
|---|---|---:|---|
| `/simple` | GET | 100 | Preprost odgovor JSON |
| `/compute` | GET | 10 | Procesorski izračuni |
| `/file-read` | GET | 20 | Branje datoteke |
| `/file-write` | POST | 20 | Pisanje datoteke |
| `/json` | POST | 20 | Obdelava podatkov JSON |
| `/auth` | POST | 10 | Preverjanje gesla z bcryptjs |

Vsaka kombinacija scenarija in okolja ima 25 ponovitev. Posamezna meritev vključuje 5 sekund ogrevanja, 2 sekundi premora, 30 sekund merjenja in 35 sekund premora pred naslednjo meritvijo.

### Zahteve

- Windows PowerShell ali PowerShell 7
- Docker Desktop
- Node.js in npm

### Zagon

```powershell
.\run-load-benchmarks.ps1
```

Zagon samo scenarija `/simple` s petimi ponovitvami:

```powershell
.\run-load-benchmarks.ps1 -Scenarios simple -Repetitions 5
```

Ročni zagon aplikacij:

```powershell
docker compose build
docker compose up -d
```

- Node.js: `http://localhost:3000`
- Bun.js: `http://localhost:3001`

Ustavitev:

```powershell
docker compose down
```

Rezultati se shranijo v časovno označeno podmapo znotraj `results`. Končna serija meritev je v `results/load-tests-20260904-052953`.

## Čas zagona aplikacije

Meritve so bile izvedene ločeno v Fedora 44 VM, brez Dockerja. Uporabljeni so bili Node.js 22.23.1, Bun.js 1.3.14 in Python 3.14.3.

Skripta izvede 5 poskusnih zagonov za vsako okolje in 100 parnih meritev. Vrstni red Node.js in Bun.js naključno premeša. Med zagoni uporabi 0,20 sekunde premora.

Zagon iz mape `startup-benchmark`:

```bash
python3 benchmark.py
```

Rezultati se shranijo v `startup-benchmark/results`:

- `startup_measurements.csv`
- `startup_summary.csv`
- `test_environment.txt`

## Rezultati obremenitvenih testov

Mapa `results/load-tests-20260904-052953` vsebuje:

- surove rezultate Autocannona
- meritve `docker stats`
- vrstni red izvedb
- povzetke rezultatov

Surovih rezultatov ne spreminjamo. Meritve časa zagona in obremenitveni testi so bili izvedeni v različnih okoljih, zato jih primerjamo samo znotraj iste vrste testa.

## Avtor

Miha Blatnik  
Univerza v Mariboru, Fakulteta za elektrotehniko, računalništvo in informatiko
