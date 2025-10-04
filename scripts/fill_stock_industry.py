#!/usr/bin/env python3
"""
股票行业信息填充工具
根据A股列表.xlsx中的股票行业信息，填充到MySQL数据库的stocks表中
"""

import pandas as pd
import os
import sys
from datetime import datetime
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockIndustryFiller:
    def __init__(self, excel_path):
        """
        初始化填充器
        
        Args:
            excel_path: Excel文件路径
        """
        self.excel_path = excel_path
        self.db_config = self._get_db_config()
        
    def _get_db_config(self):
        """获取数据库配置"""
        try:
            # 尝试使用项目中的数据库配置管理器
            from src.data.db.database_config import get_database_config
            config_manager = get_database_config()
            # 通过get_config()方法获取配置字典
            db_config = config_manager.get_config()
            return {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 3306),
                'user': db_config.get('user', 'stock_user'),
                'password': db_config.get('password', 'stock_password'),
                'database': db_config.get('database', 'stock_evaluation'),
                'charset': 'utf8mb4'
            }
        except ImportError as e:
            logger.warning(f"无法导入数据库配置管理器: {e}")
            # 使用默认配置
            return {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'user': os.getenv('DB_USER', 'stock_user'),
                'password': os.getenv('DB_PASSWORD', 'stock_password'),
                'database': os.getenv('DB_NAME', 'stock_evaluation'),
                'charset': 'utf8mb4'
            }
    
    def read_excel_data(self):
        """
        读取Excel文件数据
        
        Returns:
            DataFrame: 包含股票名称和行业信息的数据框
        """
        try:
            logger.info(f"开始读取Excel文件: {self.excel_path}")
            
            # 读取Excel文件
            df = pd.read_excel(self.excel_path)
            
            # 打印列名以便调试
            logger.info(f"Excel文件列名: {df.columns.tolist()}")
            logger.info(f"数据形状: {df.shape}")
            
            # 查找包含股票名称和行业的列
            stock_name_col = None
            industry_col = None
            
            # 可能的股票名称列名
            stock_name_keywords = ['股票名称', '股票名', '名称', 'name', 'stock_name']
            # 可能的行业列名
            industry_keywords = ['行业', '行业分类', 'industry', 'sector']
            
            for col in df.columns:
                col_lower = str(col).lower()
                for keyword in stock_name_keywords:
                    if keyword in col_lower:
                        stock_name_col = col
                        break
                for keyword in industry_keywords:
                    if keyword in col_lower:
                        industry_col = col
                        break
            
            if stock_name_col is None:
                # 如果没有找到明确的股票名称列，使用第一列
                stock_name_col = df.columns[0]
                logger.warning(f"未找到明确的股票名称列，使用第一列: {stock_name_col}")
            
            if industry_col is None:
                # 如果没有找到明确的行业列，尝试使用第二列
                if len(df.columns) > 1:
                    industry_col = df.columns[1]
                    logger.warning(f"未找到明确的行业列，使用第二列: {industry_col}")
                else:
                    raise ValueError("Excel文件中没有找到行业列")
            
            logger.info(f"使用股票名称列: {stock_name_col}")
            logger.info(f"使用行业列: {industry_col}")
            
            # 提取需要的列
            result_df = df[[stock_name_col, industry_col]].copy()
            result_df.columns = ['stock_name', 'industry']
            
            # 去除行业字段的首字母（如果存在）
            result_df['industry'] = result_df['industry'].astype(str).apply(
                lambda x: x[1:] if x and len(x) > 1 else x
            )
            
            # 去除空值
            result_df = result_df.dropna()
            
            logger.info(f"成功读取 {len(result_df)} 条股票行业数据")
            logger.info(f"前5条数据:\n{result_df.head()}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"读取Excel文件失败: {e}")
            raise
    
    def connect_database(self):
        """连接MySQL数据库"""
        try:
            import mysql.connector
            connection = mysql.connector.connect(**self.db_config)
            logger.info("成功连接数据库")
            return connection
        except Exception as e:
            logger.error(f"连接数据库失败: {e}")
            raise
    
    def fill_industry_data(self, batch_size=100):
        """
        填充行业数据到数据库
        
        Args:
            batch_size: 批量处理大小
        """
        # 读取Excel数据
        industry_data = self.read_excel_data()
        
        # 连接数据库
        connection = self.connect_database()
        
        try:
            with connection.cursor() as cursor:
                # 检查stocks表是否存在industry字段
                cursor.execute("""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = %s 
                    AND TABLE_NAME = 'stocks' 
                    AND COLUMN_NAME = 'industry'
                """, (self.db_config['database'],))
                
                if not cursor.fetchone():
                    # 如果industry字段不存在，则添加
                    logger.info("stocks表中不存在industry字段，正在添加...")
                    cursor.execute("ALTER TABLE stocks ADD COLUMN industry VARCHAR(100)")
                    connection.commit()
                    logger.info("成功添加industry字段")
                
                # 批量更新行业信息
                total_rows = len(industry_data)
                updated_count = 0
                
                for i in range(0, total_rows, batch_size):
                    batch = industry_data.iloc[i:i+batch_size]
                    
                    for _, row in batch.iterrows():
                        stock_name = row['stock_name']
                        industry = row['industry']
                        
                        # 更新数据库
                        cursor.execute("""
                            UPDATE stocks 
                            SET industry = %s 
                            WHERE name = %s
                        """, (industry, stock_name))
                        
                        updated_count += cursor.rowcount
                    
                    connection.commit()
                    logger.info(f"已处理 {min(i + batch_size, total_rows)}/{total_rows} 条数据")
                
                logger.info(f"成功更新 {updated_count} 条股票的行业信息")
                
        except Exception as e:
            logger.error(f"填充行业数据失败: {e}")
            connection.rollback()
            raise
        finally:
            connection.close()
    
    def validate_results(self):
        """验证填充结果"""
        connection = self.connect_database()
        
        try:
            with connection.cursor() as cursor:
                # 统计有行业信息的股票数量
                cursor.execute("SELECT COUNT(*) FROM stocks WHERE industry IS NOT NULL")
                with_industry = cursor.fetchone()[0]
                
                # 统计总股票数量
                cursor.execute("SELECT COUNT(*) FROM stocks")
                total_stocks = cursor.fetchone()[0]
                
                # 统计不同行业的数量
                cursor.execute("SELECT industry, COUNT(*) FROM stocks WHERE industry IS NOT NULL GROUP BY industry")
                industry_stats = cursor.fetchall()
                
                logger.info("=== 验证结果 ===")
                logger.info(f"总股票数量: {total_stocks}")
                logger.info(f"有行业信息的股票数量: {with_industry}")
                logger.info(f"行业覆盖率: {with_industry/total_stocks*100:.2f}%")
                logger.info("行业分布统计:")
                
                for industry, count in industry_stats:
                    logger.info(f"  {industry}: {count} 只股票")
                
        except Exception as e:
            logger.error(f"验证结果失败: {e}")
            raise
        finally:
            connection.close()

def main():
    """主函数"""
    excel_path = "/Users/xieyongliang/stock-evaluation3/A股列表.xlsx"
    
    # 检查文件是否存在
    if not os.path.exists(excel_path):
        logger.error(f"Excel文件不存在: {excel_path}")
        return
    
    # 创建填充器
    filler = StockIndustryFiller(excel_path)
    
    try:
        # 填充行业数据
        logger.info("开始填充行业数据...")
        filler.fill_industry_data()
        
        # 验证结果
        logger.info("开始验证填充结果...")
        filler.validate_results()
        
        logger.info("行业信息填充完成!")
        
    except Exception as e:
        logger.error(f"执行失败: {e}")

if __name__ == "__main__":
    main()