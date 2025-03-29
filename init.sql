CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE vectorstores (
    vectorstore_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

CREATE TABLE documents (
    doc_id SERIAL PRIMARY KEY,
    vectorstore_id INTEGER REFERENCES vectorstores(vectorstore_id),
    content TEXT,
    doc_metadata JSONB,
    embedding vector(768), 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
