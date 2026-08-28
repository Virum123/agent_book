from dotenv import load_dotenv

load_dotenv()


from opensearchpy import OpenSearch
from langchain_openai import OpenAIEmbeddings


client = OpenSearch(
    hosts=[
        {
            "host": "localhost",
            "port": 9200,
        }
    ]
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


def hybrid_search(
    query: str,
    index_name: str,
    k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[dict]:
    """하이브리드 검색을 수행합니다.

    Args:
        query: 검색 질문
        index_name: 검색할 인덱스 이름
        k: 반환할 결과 수
        vector_weight: 벡터 검색 가중치(0~1)
        keyword_weight: 키워드 검색 가중치(0~1)

    Returns:
        검색 결과 리스트
    """

    # 질문을 벡터로 변환
    query_vector = embeddings.embed_query(query)

    # 하이브리드 검색 쿼리(RRF 방식)
    search_query = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    # 벡터 검색(의미 기반)
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k * 2,  # 더 많이 가져와서 재순위
                            }
                        }
                    },

                    # 키워드 검색(BM25)
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "boost": 1.0,
                            }
                        }
                    },
                ]
            }
        },

        # score-based weighted fusion으로 결과 병합
        "search_pipeline": {
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {
                            "technique": "min_max"
                        },
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {
                                "weights": [
                                    vector_weight,
                                    keyword_weight,
                                ]
                            },
                        },
                    }
                }
            ]
        },
    }

    response = client.search(
        index=index_name,
        body=search_query,
    )

    results = []

    for hit in response["hits"]["hits"]:
        results.append(
            {
                "content": hit["_source"]["content"],
                "score": hit["_score"],
                "metadata": hit["_source"].get(
                    "metadata",
                    {},
                ),
            }
        )

    return results

if __name__ == "__main__":
    results = hybrid_search(
        query="육아휴직 기간은 얼마나 되나요?",
        index_name="company-docs",
        k=5,
        vector_weight=0.7,
        keyword_weight=0.3,
    )

    for i, result in enumerate(results, start=1):
        print(f"\n[{i}위]")
        print(f"점수: {result['score']}")
        print(f"메타데이터: {result['metadata']}")
        print(f"내용:\n{result['content']}")