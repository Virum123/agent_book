from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = [
    """
    # 넥스트랩 휴가 정책(2026년 개정)
    ## 1. 연차휴가
    ### 1.1 부여 기준
    - 입사 1년 미만: 매월 1일씩 부여(최대 11일)
    - 입사 1년 이상: 연 15일 일괄 부여
    - 3년 이상 근속 시 매 2년마다 1일 추가(최대 25일)

    ### 1.2 사용 방법
    - HR 시스템에서 최소 3일 전 신청
    - 반차(0.5일) 단위로 사용 가능
    - 미사용 연차는 다음 해로 이월 불가(단, 연말 5일까지 이월 가능)

    ## 2. 특별휴가

    ### 2.1 경조사 휴가
    - 본인 결혼: 5일
    - 자녀 결혼: 1일
    - 배우자/본인 부모 사망: 5일
    - 조부모/형제자매 사망: 3일

    ### 2.2 출산/육아 휴가
    - 출산휴가: 90일(산전후휴가)
    - 배우자 출산휴가: 10일
    - 육아휴직: 최대 1년(만 8세 이하 자녀)

    ## 3. 병가
    - 유급 병가: 연 60일
    - 무급 병가: 연 30일 추가 가능
    """
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, # 각 청크의 최대 글자 수
    chunk_overlap=20, # 청크 간 겹치는 글자 수
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        " ", # 분할 기준
    ],
)

small_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # 각 청크의 최대 글자 수
    chunk_overlap=20, # 청크 간 겹치는 글자 수
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        " ", # 분할 기준
    ],
)

chunks = text_splitter.create_documents(documents)
print(f"총{len(chunks)}개의 청크로 분할됨")

# 첫 번째 청크 확인

print(f"\n[청크1]\n{chunks[0].page_content[:200]}...")

s_chunks = small_text_splitter.create_documents(documents)
print(f"총{len(s_chunks)}개의 청크로 분할됨")

# 첫 번째 청크 확인

print(f"\n[청크1]\n{s_chunks[0].page_content[:200]}...")

from opensearchpy import OpenSearch
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# OpenSearch 클라이언트 설정
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=None,  # 개발 환경에서는 인증 비활성화
    use_ssl=False,
)

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 인덱스 생성(벡터 검색 설정 포함)
index_name = "company-docs"

index_body = {
    "settings": {
        "index": {
            "knn": True,  # 벡터 검색 활성화
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text"
            },  # 문서 내용(키워드 검색용)

            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,  # text-embedding-3-small 차원
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",  # 코사인 유사도
                    "engine": "nmslib"
                }
            },

            "metadata": {
                "type": "object"
            },  # 추가 메타데이터
        }
    }
}

# 기존 인덱스가 있으면 삭제 후 생성
if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)

client.indices.create(
    index=index_name,
    body=index_body,
)

print(f"인덱스 '{index_name}' 생성 완료")

# 청크를 임베딩하고 인덱싱
for i, chunk in enumerate(chunks):
    # 텍스트를 벡터로 변환
    vector = embeddings.embed_query(
        chunk.page_content
    )

    # OpenSearch에 저장
    doc = {
        "content": chunk.page_content,
        "embedding": vector,
        "metadata": {
            "source": "휴가정책.md",
            "chunk_index": i,
        }
    }

    client.index(
        index=index_name,
        body=doc,
        id=str(i),
    )

# 인덱스 새로고침(검색 가능하도록)
client.indices.refresh(index=index_name)

print(f"{len(chunks)}개 문서 인덱싱 완료")



# =====================
# 벡터 검색
# =====================
def search_documents(query: str, k: int = 3) -> list[dict]:
    """질문과 유사한 문서를 검색합니다."""

    # 질문을 벡터로 변환
    query_vector = embeddings.embed_query(query)

    # 벡터 검색 쿼리
    search_query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        }
    }

    # 검색 실행
    response = client.search(index=index_name,body=search_query)

    # 결과 추출
    results = []

    for hit in response["hits"]["hits"]:
        results.append({
            "content": hit["_source"]["content"],
            "score": hit["_score"],
            "metadata": hit["_source"]["metadata"],
        })

    return results


# ========================
# RAG 파이프라인 구현
# ========================

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# LLM 설정
llm = init_chat_model(
    model_provider="openai",
    model="gpt-5.2",
    temperature=0,
)

# RAG 프롬프트 템플릿
# 4장에서 배운 프롬프트 엔지니어링 원칙이 RAG에도 그대로 적용됩니다.
# - 역할 정의: "회사 정책 안내 도우미"
# - 명확한 제약: "주어진 문서만 참고"
# - 구체적 지침: 3가지 규칙으로 행동 명시
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 회사 정책 안내 도우미입니다.
주어진 문서만을 참고하여 질문에 답변하세요.

규칙:

1. 문서에 있는 내용만 답변하세요.

2. 문서에 없는 내용은 "해당 정보는 문서에서 확인할 수 없습니다."라고 답변하세요.

3. 답변 끝에 참고한 문서 출처를 명시하세요."""),
    ( "user", """[참고 문서]
{context}

[질문]
{question}"""),
])


def ask_rag(question: str) -> str:
    """RAG를 사용해 질문에 답변합니다."""

    # 1. 관련 문서 검색
    search_results = search_documents(
        question,
        k=3,
    )

    # 2. 검색 결과를 컨텍스트로 구성
    context = "\n\n---\n\n".join([
        f"[출처: {r['metadata']['source']}, 청크 {r['metadata']['chunk_index']}]\n{r['content']}"
        for r in search_results
    ])

    # 3. LLM에 질문과 컨텍스트 전달
    chain = rag_prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response.text

# 테스트
questions = [
    "신입사원 연차는 며칠인가요?",
    "배우자 출산휴가는 며칠인가요?",
    "연차 이월이 가능한가요?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask_rag(q)}")
    print("-" * 50)