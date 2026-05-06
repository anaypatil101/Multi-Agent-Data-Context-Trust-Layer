-- E-commerce sample schema.
-- Includes explicit FKs, intentionally ambiguous columns (data, val, misc_flag),
-- and type-name mismatches (email stored as INTEGER) to exercise trust scoring.

CREATE TABLE users (
    id          INTEGER     NOT NULL PRIMARY KEY,
    email       VARCHAR(255) NOT NULL,
    name        VARCHAR(128) NOT NULL,
    phone       VARCHAR(20),
    created_at  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status      VARCHAR(20) NOT NULL DEFAULT 'active',
    data        TEXT
);

CREATE TABLE categories (
    id          INTEGER     NOT NULL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    parent_id   INTEGER,
    description TEXT,
    FOREIGN KEY (parent_id) REFERENCES categories(id)
);

CREATE TABLE products (
    id          INTEGER     NOT NULL PRIMARY KEY,
    sku         VARCHAR(50) NOT NULL,
    name        VARCHAR(255) NOT NULL,
    description TEXT,
    price       DECIMAL(10,2) NOT NULL,
    category_id INTEGER     NOT NULL,
    created_at  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    val         INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE orders (
    id          INTEGER     NOT NULL PRIMARY KEY,
    user_id     INTEGER     NOT NULL,
    order_date  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total       DECIMAL(10,2) NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    misc_flag   BOOLEAN     DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id          INTEGER     NOT NULL PRIMARY KEY,
    order_id    INTEGER     NOT NULL,
    product_id  INTEGER     NOT NULL,
    quantity    INTEGER     NOT NULL DEFAULT 1,
    unit_price  DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (order_id)   REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE payments (
    id              INTEGER     NOT NULL PRIMARY KEY,
    order_id        INTEGER     NOT NULL,
    amount          DECIMAL(10,2) NOT NULL,
    method          VARCHAR(30) NOT NULL,
    paid_at         TIMESTAMP,
    confirmation_id VARCHAR(100),
    x               INTEGER,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);
