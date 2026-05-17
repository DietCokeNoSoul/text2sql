-- 初始化测试数据库的 SQL 脚本

-- 创建 tb_shop 表
CREATE TABLE tb_shop (
    id INT PRIMARY KEY,
    name VARCHAR(255)
);

-- 创建 tb_user 表
CREATE TABLE tb_user (
    id INT PRIMARY KEY,
    username VARCHAR(255)
);

-- 插入测试数据
INSERT INTO tb_shop (id, name) VALUES
(1, 'Shop A'),
(2, 'Shop B');

INSERT INTO tb_user (id, username) VALUES
(1, 'User1'),
(2, 'User2');