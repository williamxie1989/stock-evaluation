#!/bin/bash
# MySQL数据库和用户初始化脚本

echo "开始初始化MySQL数据库..."

# 创建开发环境数据库和用户
mysql -u root << EOF
-- 创建开发环境数据库
CREATE DATABASE IF NOT EXISTS stock_evaluation CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建开发环境用户
CREATE USER IF NOT EXISTS 'stock_user'@'localhost' IDENTIFIED BY 'stock_password';
GRANT ALL PRIVILEGES ON stock_evaluation.* TO 'stock_user'@'localhost';

-- 创建测试环境数据库
CREATE DATABASE IF NOT EXISTS stock_evaluation_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建测试环境用户
CREATE USER IF NOT EXISTS 'stock_test'@'localhost' IDENTIFIED BY 'test_password';
GRANT ALL PRIVILEGES ON stock_evaluation_test.* TO 'stock_test'@'localhost';

-- 创建生产环境数据库
CREATE DATABASE IF NOT EXISTS stock_evaluation_prod CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建生产环境用户
CREATE USER IF NOT EXISTS 'stock_prod'@'localhost' IDENTIFIED BY 'prod_password';
GRANT ALL PRIVILEGES ON stock_evaluation_prod.* TO 'stock_prod'@'localhost';

-- 刷新权限
FLUSH PRIVILEGES;

SHOW DATABASES LIKE 'stock_%';
SELECT User, Host FROM mysql.user WHERE User LIKE 'stock_%';
EOF

echo "MySQL数据库和用户初始化完成！"
echo ""
echo "创建的数据库："
echo "- stock_evaluation (开发环境)"
echo "- stock_evaluation_test (测试环境)"
echo "- stock_evaluation_prod (生产环境)"
echo ""
echo "创建的用户："
echo "- stock_user/stock_password (开发环境)"
echo "- stock_test/test_password (测试环境)"
echo "- stock_prod/prod_password (生产环境)"