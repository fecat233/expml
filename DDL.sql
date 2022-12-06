# 创建数据库
CREATE DATABASE IF NOT EXISTS expml;

# 删除数据库
DROP DATABASE IF EXISTS expml;

# 使用数据库
USE expml;

# 查询当前数据库
SELECT DATABASE();

# =======================================

# DDL表操作 查询
# 查询当前数据库所有的表
SHOW TABLES;

# 查询表结构
DESC 表名;

# 查询指定表的建表语句
SHOW CREATE TABLE 表名;