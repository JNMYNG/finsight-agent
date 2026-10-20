# Day 2-2 사업보고서 RAG

## 범위와 구성

기존 Day 1 수치 계층과 Day 2 LangGraph를 재사용한다. 기업은 삼성전자(00126380),
SK하이닉스(00164779), 사업연도는 2024, 보고서는 사업보고서(11011)로 제한한다.
원문은 전체 ZIP을 수집하지만 색인은 사업 내용(II)과 경영진단(IV) 부분으로 한정한다.
Streamlit UI와 Docker는 포함하지 않는다.

| 계층 | 파일 | 역할 |
| --- | --- | --- |
| 수집 | `dart/client.py`, `rag/service.py` | 기존 재무 API의 접수번호로 document.xml 요청 |
| 파싱 | `rag/parser.py` | 본문 선택, 기업·연도 검증, 제목·문단·표 분할 |
| 임베딩 | `rag/embeddings.py` | OpenAI text-embedding-3-small, 1536차원 |
| 색인 | `rag/index.py` | Chroma 지속 저장, SQLite 활성 세대 manifest |
| 검색 | `rag/service.py` | 메타데이터 필터, 거리·주제·문장·시점 검증 |
| Agent | `agent/tools.py`, `graph.py`, `reporting.py` | search_disclosures 실행, 수치와 근거 결합 |

## 실제 원문 처리

공식 `GET /api/document.xml`에 인증키와 접수번호를 전달하면 ZIP을 반환한다.
실제 두 ZIP에는 본문 XML과 감사보고서 XML이 각각 들어 있었다. 본문은 접수번호와 같은 이름의
`접수번호.xml`만 선택하고, 감사보고서 `_00760.xml`, `_00761.xml`은 색인하지 않는다.

본문은 UTF-8 DART 태그 문서다. 표준 XML 파서가 실제 본문에서 실패하여 BeautifulSoup의
내장 html.parser로 읽는다. 외부 선언/엔티티를 거부하고 스크립트·스타일 등은 제거한다.
파서는 네트워크 리소스를 가져오거나 문서의 지시문을 실행하지 않는다.

`COMPANY-NAME@AREGCIK`, `DOCUMENT-NAME@ACODE`, 표지의 사업연도를 요청과 대조한다.
문서 제목만 보고 다른 기업이나 연도의 메타데이터를 붙이지 않는다.
문단은 최대 1400자로 분할하고 제목 경로를 유지한다. 표는 행과 셀 구분을 보존하며 rowspan/colspan을
표시한다. 긴 행은 조각별로 같은 행 번호를 유지한다. 복잡한 병합 표를 원래 시각적 배치로 복원하지는 않는다.

조각마다 기업명·고유번호·사업연도·보고서 유형·접수번호·출처 URL·본문 파일명·제목 경로·XML 행·
블록/조각 번호·표 행 범위·원문 SHA-256·파서 버전을 저장한다. PDF 페이지 번호를 만들어 내지 않는다.

## 중복과 재색인

조각 ID는 원문 해시·기업/연도/보고서·파서 버전·분할 크기·위치에서 결정적으로 생성한다.
동일 조각은 upsert하며 이미 있는 세대의 조각은 임베딩을 재생성하지 않는다.
임베딩은 명시적으로 주입하고 Chroma 기본 임베딩 모델을 다운로드하지 않는다.
한 요청은 최대 24개 조각이며 입력마다 UTF-8 8000바이트 이하를 검사해 토큰 상한을 보수적으로 제한한다.

새 세대의 조각 저장이 모두 끝나야 SQLite manifest에서 활성 세대를 원자적으로 갱신한다.
중간 실패 시 이전 활성 세대는 유지하고 재시도 시 저장된 배치를 재사용한다.
원문·파서가 바뀌어 조각 수가 줄어도 검색은 활성 세대만 참조한다. 이전 세대는 삭제하지 않아 디스크가
증가할 수 있다. 임베딩 모델/차원이 다른 DB를 섞지 않는다.

## 검색과 근거 부족

검색은 기업 코드, 사업연도, 보고서 유형, 접수번호, 활성 세대, 문단 종류로 먼저 범위를 제한한다.
그 범위에서 코사인 거리순 후보 최대 100개를 검토하고, 최대 3개(직접 검색 시 1~5개)를 반환한다.
표도 색인하지만 이번 실적 설명 검색은 서술 문단을 사용한다. 단독 표 수치로 원인을 설명하지 않는다.

검색 임베딩에도 기업·연도·보고서 문맥을 넣는다. "실적 차이"처럼 짧은 일반 질문에는 범위 내 두 기업의
검색 주제인 메모리·HBM·제품·수요·판매가격 등의 고정 검색어를 보완한다. 이는 원인에 대한 결론이 아니다.
적용한 전체 검색어는 `retrieval_query`에 기록한다. 명시적인 원가·DX 등 별도 주제는 보존한다.

거리 0.55 이하, 주제 일치, 충분한 설명 문장을 모두 요구한다. 실적 설명에서는 짧은 소제목,
다른 연도 회고와 그 뒤의 연속 문장, 전망·계획, 연도가 불분명한 현재형 설명을 보수적으로 제외한다.
이 규칙은 제한된 한국어 판정이며 검색 정확도나 완전한 시점 해석을 보장하는 분류기는 아니다.

없거나 기준에 미달하면 `NOT_INDEXED`/`INSUFFICIENT_EVIDENCE`를 반환하고 원인을 생성하지 않는다.
임베딩 실패는 구조화된 오류로 보고한다. 최초 색인은 명시적 `rag-index` 명령으로 수행한다.

## Agent 경계

`agent --rag`에서만 search_disclosures를 활성화한다. 기존 모드는 이전 Tool 구성을 유지한다.
비교 요청은 Python 재무 Tool과 각 기업의 검색 Tool을 조합한다. 요청 기업 중 검색을 생략하면
완료로 표시하지 않는다. 연도·보고서 조건 변경도 기존 Agent 입력 검증에 연결했다.

문서 본문·제목은 Tool 선택 모델에 보내지 않는다. 모델은 신뢰된 기업 식별자·검색 성공 여부·조각 ID만
관측하며, 본문 지시문으로 다른 Tool을 호출할 경로를 만들지 않는다. 전체 원문 근거는 로컬 결과/로그에서
검토할 수 있다. 최종 보고서는 Python이 계산 결과와 검색 문장만 렌더링한다.

인용 안의 숫자는 원문 수치 생략으로 표시하고 재무 금액·비율은 기존 Python 결과로 제공한다.
원문 자체는 Tool 증거에 보존한다. 발췌 문장 사이의 `[…]`는 생략을 뜻한다.
각 근거에 기업·연도·보고서 유형·접수번호·출처 링크·본문 위치를 표시한다.
자유 형식 LLM의 인과 서술이나 실적 차이의 정량 기여 분석은 제공하지 않는다.

## 실행과 검증

```bash
uv sync --extra dev --locked
.venv/bin/python -m finsight rag-index
.venv/bin/python -m finsight rag-search 삼성전자 '메모리 HBM 수요와 수익성 개선 요인'
.venv/bin/python -m finsight rag-search SK하이닉스 '메모리 HBM 수요와 수익성 개선 요인'
.venv/bin/python -m finsight agent --rag '삼성전자와 SK하이닉스의 2024년 영업이익률을 비교하고, 두 기업의 사업보고서에서 실적 차이와 관련된 내용을 찾아 설명해 줘.'
.venv/bin/python scripts/verify_rag_live.py
.venv/bin/pytest --cov=finsight --cov-report=term-missing
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

API 키·원문·DB·로그는 Git에 포함하지 않는다. 실제 API 명령은 사용료/호출 한도를 소모할 수 있다.
pytest는 합성 문서·Mock API·테스트 임베딩을 사용하지만, 필터/재색인은 실제 로컬 Chroma로 검증한다.
실제 수집·OpenAI 임베딩·검색·Agent 검증 결과는 [decisions.md](decisions.md)에 별도로 기록했다.

## 참고

- [OpenDART 공시서류 원본](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019003)
- [OpenAI Embeddings](https://developers.openai.com/api/docs/guides/embeddings)
- [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
