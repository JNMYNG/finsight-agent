# FinSight

OpenDART 실제 재무제표를 조회하고, Python으로 수치를 계산·비교하는 금융 분석 프로젝트입니다.
Day 1의 데이터 조회·정규화·비교 계층 위에 Day 2 LangGraph Agent를 구현했습니다.
자연어 질문에 따라 OpenAI가 Tool을 선택하고, Python이 재무 수치를 계산하고 보고서를 작성합니다.
Day 2-2에서는 실제 2024년 사업보고서의 OpenAI Embeddings·ChromaDB 검색을 연결했습니다.
Day 3에서는 기존 Agent를 사용하는 로컬 Streamlit 대시보드를 추가했습니다.
FastAPI, Docker, 클라우드 배포는 구현하지 않았습니다.

## 구조

```text
.
├── pyproject.toml / uv.lock     # 패키지 설정과 고정된 의존성
├── .python-version             # Python 3.11
├── .env.example
├── streamlit_app.py            # 로컬 웹 앱 진입점
├── .streamlit/config.toml      # 화면 테마·로컬 바인딩·통계 수집 비활성화
├── src/finsight/
│   ├── config.py               # 명시적인 환경변수 로딩
│   ├── models.py               # 기업·재무항목·근거·비교 결과 모델
│   ├── errors.py               # 설정·통신·API·데이터 오류
│   ├── dart/
│   │   ├── client.py           # HTTP, API status, ZIP 오류 처리
│   │   └── corporations.py     # corpCode.xml 기업명·종목코드 인덱스
│   ├── financials.py           # 순수 함수: 항목 추출과 Decimal 계산
│   ├── services.py             # 조회 흐름과 두 기업 비교
│   ├── serialization.py        # Decimal·근거 모델의 공통 JSON 변환
│   ├── agent/
│   │   ├── graph.py            # LangGraph 선택·실행·보고서 노드
│   │   ├── tools.py            # 입력 검증과 Day 1 서비스 어댑터
│   │   ├── model.py            # OpenAI SDK 및 주입 가능한 모델 인터페이스
│   │   ├── reporting.py        # Tool 근거만 사용하는 Python 보고서
│   │   └── audit.py            # 키를 제거한 JSONL 호출 로그
│   ├── rag/                   # 원문 모델·파서·임베딩·Chroma 색인·검색 서비스
│   ├── web/                   # 조건 검증·Agent 런타임·표/차트/근거 표시
│   ├── cli.py                  # Day 1 명령 + agent 명령
│   └── __main__.py
├── tests/                      # 실제 키·네트워크를 사용하지 않는 테스트
├── scripts/verify_live.py       # 명시적으로 실행하는 복수 연도 실제 API 검증
└── docs/                       # decisions.md, agent.md, rag.md, web.md
```

기존 [FinAgent](https://github.com/JNMYNG/FinAgent)의 `agent.py`와 `tools.py`에서
Agent와 Python 계산 도구를 분리한 방식을 참고했습니다. 이 저장소에서는 모델·통신·계산·서비스를
분리하여 다음 단계에서 도구 어댑터를 붙일 수 있도록 구성했습니다.

## 설치와 설정

Python 3.11과 uv가 있다면:

```bash
uv sync --extra dev --locked
```

또는 pip로 설치할 수 있습니다. 이 방법은 `pyproject.toml`의 버전 범위를 사용하며,
`uv.lock`에 고정된 환경을 그대로 재현하려면 위 uv 명령을 사용합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

기존 `.env`가 없을 때만 `.env.example`을 복사하고 발급받은 키를 입력하세요.

```dotenv
OPENDART_API_KEY=발급받은_40자리_OpenDART_인증키
```

기존 설정과의 호환성을 위해 `DART_API_KEY`도 지원합니다. 두 이름이 모두 있으면
`OPENDART_API_KEY`를 우선합니다. 같은 이름의 셸 환경변수는 `.env`보다 우선합니다.
기존 `.env`를 덮어쓰지 않으며, `.env`와 `.env.*`는 Git에서 제외하고 `.env.example`만 추적합니다.
`OPENAI_API_KEY`는 Day 1 명령에 필요하지 않습니다. Day 2 Agent에는 아래 설정도 필요합니다.

```dotenv
OPENAI_API_KEY=발급받은_OpenAI_인증키
OPENAI_MODEL=gpt-4.1-mini
```

기본 모델은 실제 연동으로 검증한 `gpt-4.1-mini`이며, 다른 Tool Calling 지원 모델은
`OPENAI_MODEL`로 지정할 수 있습니다. 다른 모델과의 호환성은 별도 검증이 필요합니다.

## 로컬 웹 앱 (Day 3)

프로젝트 루트에서 실행하세요. 위의 두 API 키가 필요합니다.

```bash
uv sync --extra dev --locked
# RAG를 처음 사용하는 경우에만 실제 원문 수집·색인 (API 비용 발생)
.venv/bin/python -m finsight rag-index
.venv/bin/streamlit run streamlit_app.py
```

브라우저에서 `http://127.0.0.1:8501`을 엽니다. 기업·사업연도(최대 3개)·연결/별도를 선택하고
자연어 질문을 입력한 뒤 **분석 실행**을 누르세요. 재무 카드·비교표·연도별 차트, 공시 발췌·위치·링크,
실제 Tool 호출 순서를 표시합니다. 최초 색인은 UI에서 자동 실행하지 않습니다.

UI는 삼성전자·SK하이닉스의 연간 재무 조회를 지원합니다. RAG는 두 기업의 **2024년만** 지원합니다.
다른 연도의 재무만 조회하려면 RAG를 해제하세요. 수치가 없으면 0으로 대체하지 않습니다.
연도별 차트는 실제 조회된 값만 사용하며, 선택한 연도 중 빠진 결과는 일부 성공으로 표시합니다.
API 실패·누락·근거 부족을 정상 완료로 표시하지 않습니다. 질문의 명시 조건과 화면 선택이 다르면
수정을 안내하며 Agent가 선택 범위를 벗어나는 요청을 실행하지 못하도록 검사합니다.

실제 외부 API를 사용하는 UI 검증은 pytest와 분리합니다 (API 비용 발생).

```bash
.venv/bin/python scripts/verify_web_live.py
.venv/bin/python scripts/verify_web_live.py --years 2024 --basis OFS --no-rag
```

실행·지원 범위·오류 안내와 검증 한계는 [웹 앱 가이드](docs/web.md)를 확인하세요.

## 자연어 Agent (Day 2)

```bash
.venv/bin/python -m finsight agent '삼성전자 2024년 연결 영업이익률을 계산해 줘.'
.venv/bin/python -m finsight agent '삼성전자와 SK하이닉스의 2023년과 2024년 연결 재무정보를 비교해 줘.'
```

`model → tools → model`을 반복한 뒤 `report` 노드가 Tool 결과로 답변을 작성합니다.
기업 조회·재무제표 조회·영업이익률 계산·기업 비교 Tool과 추가 입력 요청 Tool을 제공합니다.
자유 형식 LLM 응답의 수치는 최종 보고서에 사용하지 않습니다.

결과 JSON에는 `status`, `answer`, `tool_calls`, `stop_reason`, `run_id`, `error`가 있습니다.
각 호출의 입력·실행 결과·순서를 JSON과 기본 `logs/agent.jsonl`에서 확인할 수 있습니다.
`--log-file`, `--max-tool-calls`(기본 8), `--max-model-turns`(기본 10)를 지정할 수 있습니다.
로그 파일은 Git에서 제외합니다. OpenAI 요청에는 API 사용료가 발생합니다.

검증된 결과는 `completed`, 누락·경고·일부 실패는 `partial`, 실행 불가는 `error`,
추가 조건 요청은 `needs_input`, 반복 제한 도달은 `limit_reached`입니다.
`completed`/`needs_input`은 종료 코드 0, `partial`/`error`/`limit_reached`는 1입니다.
사업연도 없이 질문하면 연도를 임의 선택하지 않고 추가 입력을 요청합니다.
자세한 구조와 한계는 [Agent 가이드](docs/agent.md)를 확인하세요.

## 사업보고서 RAG (Day 2-2)

삼성전자·SK하이닉스의 2024년 사업보고서만 지원합니다. 실제 원문 ZIP을 수집하고,
`II. 사업의 내용`, `IV. 이사의 경영진단 및 분석의견`을 색인합니다. 보고서 전체를 색인한 것은 아닙니다.

```bash
# 실제 수집·임베딩·로컬 DB 생성 (두 기업, 최초 1회)
.venv/bin/python -m finsight rag-index

# 기업·2024년·사업보고서 조건을 적용한 검색
.venv/bin/python -m finsight rag-search 삼성전자 '메모리 HBM 수요와 수익성 개선 요인'
.venv/bin/python -m finsight rag-search SK하이닉스 '메모리 HBM 수요와 수익성 개선 요인'

# 재무 비교와 원문 근거 검색을 함께 실행
.venv/bin/python -m finsight agent --rag '삼성전자와 SK하이닉스의 2024년 영업이익률을 비교하고, 두 기업의 사업보고서에서 실적 차이와 관련된 내용을 찾아 설명해 줘.'

# 실제 검색 결과를 저장된 원문·기업·연도·접수번호·위치와 대조
.venv/bin/python scripts/verify_rag_live.py
```

OpenDART 키와 OpenAI 키가 필요하며 임베딩/Agent API 사용료가 발생합니다.
원문은 `data/disclosures/`, DB는 `data/chroma/`, 실행 로그는 `logs/`에 저장하며 모두 Git에서 제외합니다.
DB 경로는 `--db-path`로 지정할 수 있습니다. 기존 `agent` 명령은 `--rag` 없이 종전 기능을 유지합니다.

동일 원문은 재색인해도 임베딩을 재생성하지 않습니다. 수정된 원문·파서 버전은 새 세대로 색인하고,
완성된 세대만 검색에 사용합니다. 이전 세대는 로컬에 보존하므로 DB 전체 레코드 수는 활성 조각 수와
다를 수 있습니다. 원문 내용은 명령으로 실행하지 않으며 모델에는 본문 대신 검색 상태와 조각 ID만 전달합니다.

설명은 검증된 관련 문장의 발췌이며, 계산한 수치는 기존 Python 결과만 사용합니다. 근거 부족·무관한 결과·
소제목·전망 문장은 실적 원인으로 사용하지 않습니다. 실적 차이의 정량적 인과 기여를 추정하지 않습니다.
구현·검증 범위와 검색 기준은 [RAG 가이드](docs/rag.md)에 정리했습니다.

## 실행

프로젝트 루트에서 실행합니다. 성공 시 stdout에 JSON, 실패 시 stderr에 오류 JSON을 출력하고
종료 코드 1을 반환합니다. 잘못된 CLI 인자는 종료 코드 2입니다.

```bash
# 기업 고유번호: 기업명 또는 6자리 영숫자 종목코드로 조회
.venv/bin/python -m finsight lookup 삼성전자
.venv/bin/python -m finsight lookup 000660

# 단일 기업: 기본값은 사업보고서, 연결 우선
.venv/bin/python -m finsight financials 삼성전자 --year 2024

# 삼성전자(왼쪽)와 SK하이닉스(오른쪽), 연결재무제표를 명시
.venv/bin/python -m finsight compare --year 2024 --basis CFS

# 기업·보고서 유형·별도재무제표를 직접 지정
.venv/bin/python -m finsight compare --left 삼성전자 --right SK하이닉스 \
  --year 2024 --report 11012 --basis OFS
```

`--env-file`은 하위 명령 앞에 지정합니다.
설치 후 `finsight compare --year 2024`로도 실행할 수 있습니다.

| 보고서 | `--report` | 비교 금액 기준 |
| --- | --- | --- |
| 사업보고서 | `11011` (기본) | 당기금액 `thstrm_amount` |
| 1분기보고서 | `11013` | 당기누적금액 `thstrm_add_amount` |
| 반기보고서 | `11012` | 당기누적금액 `thstrm_add_amount` |
| 3분기보고서 | `11014` | 당기누적금액 `thstrm_add_amount` |

분·반기 누적금액이 없으면 해당 항목은 `null`입니다. 분기금액이나 전기금액으로 대체하지 않습니다.

## 수치와 근거 해석

세 항목은 `revenue`(매출액), `operating_profit`(영업이익), `net_income`(당기순이익)입니다.
금액은 API 응답 통화의 금액을 유지하며 억원/백만원 단위로 임의 환산하지 않습니다.
`Decimal`을 사용하고 JSON에서는 정밀도를 보존하기 위해 숫자 문자열로 출력합니다.

영업이익률은 `영업이익 / 매출액 × 100`을 소수점 둘째 자리로 반올림합니다.
매출액이 0 이하이거나 입력 누락·기간/통화 불일치가 있으면 `null`과 사유를 반환합니다.
영업손실에 따른 음수 이익률은 허용합니다.

각 기업 결과는 기업명·고유번호·종목코드·사업연도·보고서 코드/이름·재무제표 구분·조회시각·API 출처를
포함합니다. 항목의 `evidence`에는 원본 계정명·금액 문자열·선택한 금액 필드·통화·당기명/일자·접수번호를
보존합니다. 접수번호는 `https://dart.fss.or.kr/dsaf001/main.do?rcpNo=접수번호`로 확인할 수 있습니다.
인증키는 결과에 포함하지 않습니다.

`difference`는 왼쪽 값에서 오른쪽 값을 뺀 값입니다. 영업이익률 자체는 `%`,
두 기업 이익률의 차이는 `percentage_points`(%p)입니다. 보고서·재무제표 구분·기간·통화가 다르면
각 값은 유지하되 차이는 `null`과 사유를 반환합니다.

## 누락·중복·실패 처리

- `--basis` 생략 시 응답에 CFS가 있으면 연결만 선택합니다. CFS가 전혀 없을 때만 OFS로 전환하고
  `FALLBACK_TO_OFS`를 기록합니다. `--basis CFS/OFS`는 해당 구분만 허용합니다.
- 연결 항목 누락을 별도 수치로 보충하지 않습니다. 공백·`-` 등은 0이 아닌 결측입니다.
- 공백·유니코드·계정 번호를 정규화하고 명시적인 계정 별칭으로 매칭합니다. 지배주주 귀속 순이익은
  전체 당기순이익의 대체 항목으로 사용하지 않습니다.
- 동일한 값·통화·기간·접수번호의 중복은 합산하지 않고 하나의 값과 모든 근거를 보존합니다.
  충돌하는 중복은 `AMBIGUOUS_ACCOUNT`와 `null`로 반환합니다.
- API 상태 오류·HTTP 실패·타임아웃·손상된 응답은 예외로 전달합니다. `013`은 `NoDataError`입니다.
  실패한 기업의 재무정보를 만들어 비교 결과를 반환하지 않습니다.

알 수 없는 계정 표기는 `ACCOUNT_MISSING`으로 남깁니다. 별칭 목록의 포괄성, 다른 결산월 기업의
누적기간 정합성, 다양한 실데이터 형태는 추가 검증이 필요합니다. Day 1에는 자동 재시도·영구 캐시·
정정공시 이력 수집을 구현하지 않았습니다. 기업목록은 서비스 인스턴스 안에서 한 번만 다운로드합니다.

## 테스트

```bash
.venv/bin/pytest --cov=finsight --cov-report=term-missing
.venv/bin/ruff check src tests scripts streamlit_app.py
.venv/bin/ruff format --check src tests scripts streamlit_app.py
```

`httpx.MockTransport`로 ZIP/XML과 JSON 응답, API 오류, 통신 실패를 검증합니다.
테스트의 재무 수치는 합성 데이터이며 실제 기업 실적을 의미하지 않습니다.
모든 테스트는 임시 디렉터리에서 실행되어 실제 `.env`를 읽지 않습니다.
Mock 검증과 실제 API 조회 결과는 [설계·검증 기록](docs/decisions.md)에 구분해서 기록합니다.

실제 API 검증은 pytest와 분리하여 명시적으로 실행합니다. 이 명령은 OpenDART 키만 사용합니다.

```bash
.venv/bin/python scripts/verify_live.py --years 2022 2023 2024
```

2022·2023·2024년 두 기업의 연결 사업보고서 6건과, 실제 OpenAI Agent의 복수 연도 비교 및
기업 조회 → 재무제표 조회 → 계산 Tool 선택을 확인했습니다. 실제 API 검증은 해당 범위에 한정됩니다.

## 공식 문서

- [기업 고유번호: corpCode.xml](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018)
- [단일회사 주요계정: fnlttSinglAcnt](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019016)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Streamlit 차트](https://docs.streamlit.io/develop/api-reference/charts/st.line_chart)

주요계정 API에는 `fs_div` 요청 인자가 없으며 응답의 구분으로 선택합니다.
공시 정정에 따라 같은 사업연도의 조회 값도 바뀔 수 있으므로 조회시각과 접수번호를 함께 확인하세요.
