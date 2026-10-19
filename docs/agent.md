# Day 2 Agent 구조와 사용법

## 실행 흐름

Day 1 `CorporationDirectory`와 `FinancialService`를 재사용한다. 새로운 재무 계산 로직이나
두 번째 OpenDART 클라이언트를 만들지 않았다. `calculate_metrics`는 Day 1 재무 요약에 있는
`calculate_operating_margin`의 계산 결과를 반환하며, 금액을 모델 입력으로 받지 않는다.

LangGraph `StateGraph`는 `START → model → tools → model`을 반복하고, 모델이 호출을 마치거나
중단 조건을 만나면 `report → END`로 종료한다. State에는 메시지, 호출 결과, 실행 순서, 종료 사유가 있다.
인증키는 State·Tool 스키마·Tool 인자에 넣지 않는다.

| Tool | 입력 | 재사용 기능 |
| --- | --- | --- |
| `lookup_company` | 기업명/종목코드 | `CorporationDirectory.find` |
| `get_financials` | 기업, 사업연도, 보고서 코드, 재무제표 구분 | `FinancialService.get_financials` |
| `calculate_metrics` | 위와 동일, 금액 입력 없음 | Day 1 조회 및 Decimal 영업이익률 계산 |
| `compare_companies` | 두 기업, 사업연도, 보고서 코드, 재무제표 구분 | Day 1 조회 및 `compare_summaries` |
| `request_clarification` | 필요한 입력 필드 목록 | 고정된 한국어 추가 입력 안내 |

OpenAI 공식 SDK를 모델 어댑터에만 사용한다. `ToolSelectingModel`을 주입할 수 있어 테스트에서
외부 모델을 부르지 않고 실제 LangGraph를 실행한다. 병렬 Tool 선택을 비활성화하고,
여러 호출이 반환되어도 순서대로 실행한다. 그래프와 재무 요약 캐시는 매 `run()`마다 새로 만든다.
기업목록 캐시는 Day 1 서비스 인스턴스의 기존 수명을 따른다.

## 수치의 신뢰 경계

모델은 Tool 종류와 기업·사업연도·보고서·연결/별도 조건을 선택한다. 서버는 strict Pydantic
입력 검증으로 타입, 범위, 추가 필드를 검사한다. 별도 요구는 OFS, 연결 요구는 CFS로 유지한다.
미지정은 null이고 Day 1의 연결 우선 정책을 사용한다.

추가로 질문에 명시된 연도·재무제표 구분·보고서와 다른 Tool 인자를 차단한다. 명시적 연도 및
`작년/지난해/재작년/올해/금년`은 한국 시간의 현재연도를 기준으로 해석한다. `2022~2024년`,
`2022년부터 2024년까지` 같은 범위도 허용하되 실제 실행 수는 호출 제한을 따른다.
이 검사는 한국어와 코드 표기에 대한 제한된 규칙이며 모든 자연어 의미를 검증하는 파서는 아니다.

모델이 완료 시 생성한 자유 형식 문장은 재무 보고서에 사용하지 않는다. `reporting.py`가 실제
Tool 결과의 기업·기간·구분·금액·계산 결과·접수번호·출처·조회시각만 렌더링한다.
이는 프롬프트의 금지 규칙만으로 숫자 생성을 막는 데 따르는 한계를 줄이기 위한 선택이다.
계산 가능한 이익률 차이에 대한 비교 설명도 Python으로 작성한다.

영업이익률은 Python의 Decimal 계산값이며, 순이익 등 다른 항목 누락을 추정해서 채우지 않는다.
조회에 성공했어도 결측이나 경고가 있으면 `partial`로 표시한다. 데이터가 없는 상태에서 모델이
그럴듯한 재무 수치를 말해도 최종 결과는 근거 없음으로 처리한다.

## 오류와 호출 제한

잘못된 기업명, 동명이인, API 013, 인증/한도 오류, HTTP/네트워크 실패, 응답 손상을 구조화한
Tool 결과로 반환한다. 입력 검증 오류에는 원문 입력 대신 잘못된 필드와 오류 유형을 넣는다.
모델은 실패 결과를 보고 추가 입력을 요청하거나 종료할 수 있다. 실패를 성공 수치로 바꾸지 않는다.

기본 제한은 Tool 호출 8회, 모델 호출 10회다. 실패 호출도 제한에 포함한다.
각 OpenAI 요청은 타임아웃 30초, 자동 재시도 0회다. Day 1 OpenDART 타임아웃은 그대로다.
모델 장애나 제한 도달 전 완료한 결과는 보존하고 종료 사유를 표시한다. 자동 모델 변경은 하지 않는다.

## 로그

기본 파일은 `logs/agent.jsonl`이다. 각 줄은 독립된 JSON이며 `run_id`로 한 실행을 구분한다.
시작·모델 호출·Tool 시작·Tool 완료·종료를 기록하고, Tool 완료 이벤트에는 실행 순서, 호출 ID,
검증된 입력, 결과/오류, 실행시간이 포함된다. 보고서 JSON의 `tool_calls`에도 같은 결과가 담긴다.
금융정보를 포함하는 로컬 로그이며 `logs/`는 Git에서 제외한다.

알고 있는 OpenAI/OpenDART 키와 인증 필드·키 형식을 출력 전에 제거한다. 모델의 자유 응답이나
HTTP 헤더, 인증된 URL, 비공개 추론은 로깅하지 않는다. 스키마 오류는 키를 제거한 설명으로 기록한다.

## 검증 범위와 제한

자동화 테스트는 ScriptedModel과 HTTP Mock으로 Tool 선택 결과의 라우팅, 입력 전달,
호출 순서, 실패 복구, 수치 계산, 맥락 유지, 로그, 호출 제한, 숫자 생성 차단을 검사한다.
ScriptedModel 테스트 자체가 실제 LLM의 의미 이해 정확도를 입증하는 것은 아니다.
별도로 실제 OpenAI와 OpenDART로 순차 Tool 호출과 복수 연도 비교를 실행한 기록은
[decisions.md](decisions.md)에 남겼다.

현재 대화 메모리/체크포인터는 없다. `needs_input`일 때 기업·연도를 포함한 완전한 질문으로 다시
실행한다. 여러 연도의 조회는 제공하지만 전년 대비 성장률처럼 Day 1에 없는 계산은 추가하지 않았다.
지원 재무지표는 영업이익률이다. 장문의 자유 서술 분석, RAG, 웹 UI도 이번 범위에 포함하지 않았다.

## 참고한 공식 문서

- [OpenAI Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [GPT-4.1 mini 기능](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
- [LangGraph Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
