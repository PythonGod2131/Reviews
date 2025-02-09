CREATE DATABASE chatbot_db;
USE chatbot_db;

CREATE TABLE responses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tag VARCHAR(255) UNIQUE,
    response TEXT
);

INSERT INTO responses (tag, response) VALUES
('greeting', 'Hello! How can I assist you?'),
('goodbye', 'Goodbye! Have a great day!');
