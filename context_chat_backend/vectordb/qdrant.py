from os import getenv

from dotenv import load_dotenv
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ..utils import value_of
from . import get_collection_name, get_user_id_from_collection
from .base import BaseVectorDB, DbException, MetadataFilter, TSearchDict

load_dotenv()

class_schema = {
    'properties': [
        {'name': 'text', 'type': 'text', 'description': 'The actual text'},
        {'name': 'type', 'type': 'text', 'description': 'The type of source/mimetype of file'},
        {'name': 'title', 'type': 'text', 'description': 'The name or subject of the source'},
        {'name': 'source', 'type': 'text', 'description': 'The source of the text (for files: `files__default: fileId`)'},
        {'name': 'start_index', 'type': 'integer', 'description': 'Start index of chunk'},
        {'name': 'modified', 'type': 'text', 'description': 'Last modified time of the file'},
        {'name': 'provider', 'type': 'text', 'description': 'The provider of the source'},
    ],
    'vector_size': 768,  # Adjust according to your vector size
    'distance': 'Cosine'
}

class VectorDB(BaseVectorDB):
    def __init__(self, embedding: Embeddings | None = None, **kwargs):
        try:
            client = QdrantClient(
                url=getenv('QDRANT_URL'),
                api_key=getenv('QDRANT_APIKEY')
            )
        except Exception as e:
            raise DbException('Error: Qdrant connection error') from e

        self.client = client
        self.embedding = embedding

    def get_users(self) -> list[str]:
        if not self.client:
            raise DbException('Error: Qdrant client not initialised')

        collections = self.client.get_collections().collections
        return [get_user_id_from_collection(coll.name) for coll in collections]

    def setup_schema(self, user_id: str) -> None:
        if not self.client:
            raise DbException('Error: Qdrant client not initialised')

        collection_name = get_collection_name(user_id)
        if collection_name in [c.name for c in self.client.get_collections().collections]:
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=768,  # Adjust according to your vector size
                distance=rest.Distance.COSINE
            ),
            payload_schema=class_schema['properties']
        )

    def get_user_client(
        self,
        user_id: str,
        embedding: Embeddings | None = None  # Use this embedding if not None or use global embedding
    ) -> VectorStore:
        self.setup_schema(user_id)

        qdrant_obj = VectorStore(
            client=self.client,
            collection_name=get_collection_name(user_id),
            text_key='text',
            embedding=(self.embedding or embedding),
        )
        return qdrant_obj

    def get_metadata_filter(self, filters: list[MetadataFilter]) -> dict | None:
        if len(filters) == 0:
            return None

        try:
            if len(filters) == 1:
                return {
                    'key': filters[0]['metadata_key'],
                    'match': {'value': filters[0]['values']}
                }

            return {
                'should': [{'key': f['metadata_key'], 'match': {'value': f['values']}} for f in filters]
            }
        except (KeyError, IndexError):
            return None

    def get_objects_from_metadata(
        self,
        user_id: str,
        metadata_key: str,
        values: list[str],
    ) -> TSearchDict:
        if not self.client:
            raise DbException('Error: Qdrant client not initialised')

        self.setup_schema(user_id)

        query_filter = self.get_metadata_filter([{'metadata_key': metadata_key, 'values': values}])
        if query_filter is None:
            raise DbException('Error: Qdrant metadata filter error')

        search_result = self.client.search(
            collection_name=get_collection_name(user_id),
            query_vector=None,
            query_filter=query_filter,
            limit=100  # Set appropriate limit
        )

        output = {}
        for hit in search_result:
            payload = hit.payload
            key_value = payload.get(metadata_key)
            if key_value in values:
                output[key_value] = {
                    'id': hit.id,
                    'modified': payload.get('modified'),
                }

        return output
