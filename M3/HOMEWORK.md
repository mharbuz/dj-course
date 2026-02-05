# Zadanie 1

TASK: stwórz mini-PRD + "zakoduj" go przy użyciu agentów "scaffoldujących frontendy"
1. zrób **“zgrubny” research**, co powinien zawierać moduł TMS pozwalający klientowi składać zamówienie transportowe (jakie dane musi podać, jak wygląda formularz/proces)
2. stwórz PRD w formie tekstu
3. W oparciu o wybrane LLMy/agenty (Lovable.dev, bolt.new, gemini, co kto lubi): stwórz “klikalny” frontend
(dane może mieć zahardkodowane)

WYMAGANIA:
- nie koduj samodzielnie (jak zwierzę 😂)
- niech ów frontend da się przeklikać + przechodzić między ekranami
- niech API backendowe będzie śmiało zahardkodowane
- zakres nie musi być szeroki, może się ograniczać nawet do 1 formularza - byle był naprawdę szczegółowy (najlepiej - aby miał wiele ekranów np. uzupełnianych krok po kroku)

CEL:
- uświadomić sobie, jak szybko / efektywnie / tanio można postawić "relatywnie mały frontend" przy użyciu dostępnych narzędzi
- uświadomić sobie, że punkt ciężkości (w budowaniu frontendów) się przesuwa. Gdzie? Zapraszam do dyskusji na discordzie 🤗

Konwersacja z Google Gemini: https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221Yk_8AcmZ_bqA1zTt4kiivL2qQq-J62wz%22%5D,%22action%22:%22open%22,%22userId%22:%22117651972521849216727%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing
efekt na Bolt.new: https://bolt.new/~/sb1-1zuxraa9


# Zadanie 2

**Developer Distractor Destroyer**

Folder: `M3/developer-distractor-destroyer`

Zawartość:
- kod wtyczki przeglądarkowej (stworzonej przy użyciu LLMów/agentów, rzecz jasna)
- statystyki/śledzenie: ile czasu na jakiej stronie spędzasz

TASK - zaimplementuj:
- statystyki z podziałem np. na dni, tygodnie, miesiące
- filtrowania okresów / kumulowania danych, np. statystyki tygodniowe
TASK (OPCJONALNIE):
- Import/Export statystyk do/z JSONa
- Jakie wrażenia a propos vibe codingu? Podziel się.

CEL:
- uświadomić sobie, jak łatwo stworzyć własną wtyczkę do przeglądarki, realizującą co tylko chcesz. Dopóki nie masz dedykowanego backendu, ograniczają Cię limity API przeglądarkowych (np. maksymalny rozmiar danych trzymanych w WebStorage)
- uświadomić sobie, że **CO + PO CO** robisz > (jest ważniejsze) niż **JAK** to robisz. Jeśli masz +-dokładną wizję co chcesz osiągnąć - LLMy ogarną API.

# Zadanie 3

TASK: Skonfiguruj w swoim coding agent serwer(y) MCP (jeśli jeszcze nie masz)

Zweryfikuj poprawność konfiguracji poprzez wywołanie przykładowego “tool”

PRZYKŁADY/INSPIRACJE:
- chrome devtools: https://github.com/ChromeDevTools/chrome-devtools-mcp
- postgres: https://github.com/HenkDz/postgresql-mcp-server, `@modelcontextprotocol/server-postgres`
- docker: https://github.com/QuantGeekDev/docker-mcp, lokalny folder `M3/mcp-docker-py`
- sequential thinking: https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking
- context 7 (API docs): https://mcp.context7.com/mcp
- playwright (automation/e2e tests): `@playwright/mcp`

CEL:
- skuteczne posługiwanie się MCP + umiejętność konfiguracji

# Zadanie 4

TASK: Rozbuduj `tms-data-generator` który jest oparty o golang/SQL.
- w oparciu o LLMy/agenty (tzn. nie koduj "na piechotę")
- dodaj info o **dostępności: kierowców i pojazdów**
- wymaga to dodania zarówno DDL (`CREATE TABLE …` - statyczne w pliku SQL) jak i DML (`INSERT INTO …` - dynamicznie generowane przy użyciu golanga)
- określ/zaprojektuj:
  - jak powinna wyglądać reprezentacja bazodanowa dostępności
  - ile encji trzeba dodać i jakich?
  - jakie powiązania będą miały nowo generowane rekordy z pozostałymi?

Folder: `M3/tms-data-generator`

WYZWANIE:
- prawdopodobnie nie znasz golanga, ale masz zaprojektować co ma być zrobione KONCEPCYJNIE
- jesteś pilotem, nie pasażerem

CEL:
- "odklejenie się" od warstwy konkretnej składni języka i praca na poziomie fundamentalnych building blocków (ify, pętle, obiekty, funkcje, klasy). Projektujesz fundamenty rozwiązania, odpowiadasz za zależności, za model danych, za flow przepływu informacji (np. kto pulluje, kto pushuje) itp - to wszystko są aspekty language-agnostic. LLM zaś "ubiera koncept w kod". Zwłaszcza że kod-punkt-odniesienia już ma

# Zadanie 5

TASK: stwórz subagents w Claude Code (jeśli nie masz)

SUBSKRYPCJA/$$$:
- jeśli nie masz “subskrypcji”, możesz zasilić konto w stylu prepaid sztywną kwotą, np. 5$ (stworzenie subagenta skonsumuje bardzo małą część tej kwoty)

Subagent ma mieć charakterystykę:
- Pomocnika w planowaniu zadań
- Iteracyjnie doprecyzowywać plan
- Szukać pytań/wątpliwości/problemów jakie jeszcze nie zostały zaadresowane
- Ma być sparing-partnerem w tworzeniu planu a NIE twórcą planu, który “zwalnia Cię z myślenia”

CEL:
- pierwsze szlify

## Zadanie 5.2 (dla chętnych)

Przemyśl poniższe pytania/problemy - i podziel się na discordzie przemyśleniami:
- w jaki sposób subagents jest zaimplementowany/zaprojektowany?
- w jaki sposób subagent jest wybierany?
- co trzeba by dodać w AZØRZE aby to umożliwić?

# Zadanie 6

TASK: Napisz własne MCP tools!
- Rozbuduj AZØRA (kod bazowy - prace domowe z M1, `M1/azor-chatdog-*`)
- Platforma/język - do wyboru - niezależnie od implementacji AZØRA

Tools: 
- 1-szy tool: listuje sesje/wątki w AZØZE (`~/.azor/*.json`) wraz z datą aktualizacji
- 2-gi tool: zwraca metadane + treść
- 3-ci tool: usuwa wybrany wątek/wątki

TEST:
- Prompt: “usuń wątki z ostatniej doby”. Agent/model mają zorkiestrować i wykonać całość :)
- Manual TEST: `mcp-inspector`

# Zadanie 7

TASK: rozbuduj slash command zmieniający sesję
- Rozbuduj AZØRA (kod bazowy - prace domowe z M1, `M1/azor-chatdog-*`)
- Obecnie azor wspiera komendę `/switch <session-id>` która zmienia aktualną sesję
- Zadanie polega na dodaniu terminalowego “dropdowna” który wylistuje sesje do wyboru. Wszystko keyboard-based :)
- Jeśli zaimplementowałeś/aś tytuły wątków (M2/Z6) to dropdown wyświetla tytuły
- A propos toolingu dla dropdowna - zerknij na kod obsługujący `/session`

# Zadanie 8

TASK: zaimplementuj "Doprecyzuj pytanie, użytkowniku…"
- Rozbuduj AZØRA (kod bazowy - prace domowe z M1, `M1/azor-chatdog-*`)
- Wcześniejsza praca domowa (M1/Z11) dotyczyła projektu tego rozwiązania - teraz go **implementujemy**
- Model dopytuje, kiedy uzna, że pytanie użytkownika jest niewystarczająco precyzyjne
- Rekomendowany kierunek: tool call, implementujemy funkcję 🤠 która pobiera od usera clarification
- W zależności od klienta LLM (gemini etc.) wykorzystujemy odpowiednie API

# Zadanie 9

**GTA: S2 Deliveroo to symulator kierowcy** 😎

Folder: `M3/gta-s2-deliveroo`

TASK: Zaimplementuj nową planszę/poziom:
- Parkowanie skośne (z samochodami lub bez)
- Parkowanie tyłem (obecnie gra nie weryfikuje kierunku, a jedynie czy pojazd znajduje się w całości w obszarze)
- lub gdziekolwiek poniesie Cię kreatywność 🕊️

TASK: Zaimplementuj auto do wyboru:
- Na początku gry, z jakimś nowym widokiem
- Obecnie jest auto o sztywnej charakterystyce
- Dodaj auto sportowe i ciężarówkę

CEL:
- mieć dobrą zabawę
- znaleść sweet spot - na ile powinieneś/aś mieć kontrolę nad tym, co powstaje "pod spodem"? (na ile LLM/agent mają brać odpowiedzialność za proces wytwórczy?)

# Zadanie 10

W magazynie deliveroo pojawili się źli naziści! 😱

**Symulator Magazynu Deliveroo**

Folder: `M3/warehouse-simulator`

TASK: **PORUSZ** nazistów - dosłownie i w przenośni!

Zaimplementuj wybrane ficzery:
- Niech chodzą po magazynie
- Animuj ruchy nazistów adekwatnie do pozycji względem Ciebie
- Jeśli podejdziesz blisko, niech zwrócą na Ciebie uwagę
- Niech wydają dźwięki tym głośniejsze, im bliżej jesteś

W repozytorium znajduje się masa plików:
- z grafikami, dźwiękami itp.
- `M3/warehouse-simulator/` - tu jest kod frontendowy (react/three.js, choć react jest tutaj niemalże bez znaczenia, three.js jest fundamentalny)
- `M3/wolfenstein` - galeria "podglądu" animowanych postaci w grze. Kod został "przemigrowany" przez LLM do kodu magazynu
- `M3/wolfenstein/_dev` - tu są źródłowe pliki - grafiki (sprites), dźwięki, itp. Jeśli chcesz dodawać odpowiednie dźwięki, nowe postaci itp - to pliki bierz stąd.

**Niech poniesie Cię fantazja 🕊️**
