# FinSight

OpenDART 실제 재무제표를 조회하고, Python으로 수치를 계산·비교하는 금융 분석 프로젝트입니다.
Day 1은 데이터 조회·정규화·비교 계층과 CLI, Mock 테스트까지 구현합니다.
LangGraph Agent, OpenAI 호출, RAG, ChromaDB, FastAPI, Streamlit, Docker는 후속 단계입니다.

## 구조

```text
.
├── pyproject.toml / uv.lock     # 패키지 설정과 고정된 의존성
├── .python-version             # Python 3.11
├── .env.example
├── src/finsight/
│   ├── config.py               # 명시적인 환경변수 로딩
│   ├── models.py               # 기업·재무항목·근거·비교 결과 모델
│   ├── errors.py               # 설정·통신·API·데이터 오류
│   ├── dart/
│   │   ├── client.py           # HTTP, API status, ZIP 오류 처리
│   │   └── corporations.py     # corpCode.xml 기업명·종목코드 인덱스
│   ├── financials.py           # 순수 함수: 항목 추출과 Decimal 계산
│   ├── services.py             # 조회 흐름과 두 기업 비교
│   ├── cli.py                  # JSON CLI
│   └── __main__.py
├── tests/                      # 실제 키·네트워크를 사용하지 않는 테스트
└── docs/decisions.md            # 설계 선택, 실제 오류, 검증 기록
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
`OPENAI_API_KEY`는 Day 1에 필요하지 않습니다.

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
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

`httpx.MockTransport`로 ZIP/XML과 JSON 응답, API 오류, 통신 실패를 검증합니다.
테스트의 재무 수치는 합성 데이터이며 실제 기업 실적을 의미하지 않습니다.
모든 테스트는 임시 디렉터리에서 실행되어 실제 `.env`를 읽지 않습니다.
Mock 검증과 실제 API 조회 결과는 [설계·검증 기록](docs/decisions.md)에 구분해서 기록합니다.

## 공식 문서

- [기업 고유번호: corpCode.xml](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018)
- [단일회사 주요계정: fnlttSinglAcnt](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019016)

주요계정 API에는 `fs_div` 요청 인자가 없으며 응답의 구분으로 선택합니다.
공시 정정에 따라 같은 사업연도의 조회 값도 바뀔 수 있으므로 조회시각과 접수번호를 함께 확인하세요.
