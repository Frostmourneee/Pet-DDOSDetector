import os
import tempfile
from typing import Any, Optional

import joblib
import pandas as pd
from hdfs import InsecureClient
from hdfs.util import HdfsError


class HDFSClient:
    def __init__(self, host: str = "hadoop", webhdfs_port: int = 50070) -> None:
        self.client = InsecureClient(f"http://{host}:{webhdfs_port}")
        print(f"HDFS клиент инициализирован для {host}:{webhdfs_port}")

    def read_csv(self, hdfs_path: str) -> pd.DataFrame:
        """Чтение CSV из HDFS в pandas DataFrame."""
        try:
            with self.client.read(hdfs_path) as reader:
                return pd.read_csv(reader)
        except HdfsError:
            print(f"Файл не найден в HDFS: {hdfs_path}")
            raise
        except Exception as e:
            print(f"Ошибка чтения из HDFS {hdfs_path}: {e}")
            raise

    def write_csv(self, df: pd.DataFrame, hdfs_path: str) -> None:
        """Загрузка DataFrame в HDFS как CSV."""
        tmp_file: Optional[str] = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".csv"
            )
            df.to_csv(tmp_file.name, index=False, header=False)
            tmp_file.close()

            self.client.upload(hdfs_path, tmp_file.name, overwrite=True)
            print(f"Данные сохранены в HDFS: {hdfs_path}")

        except HdfsError as e:
            print(f"Ошибка записи в HDFS {hdfs_path}: {e}")
            raise
        finally:
            if tmp_file and os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)

    def file_exists(self, hdfs_path: str) -> bool:
        """Проверка существования файла в HDFS."""
        try:
            self.client.status(hdfs_path)
            return True
        except HdfsError:
            return False

    def write_joblib(self, obj: Any, hdfs_path: str) -> None:
        """Сохранение объекта через joblib в HDFS."""
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                tmp_path = tmp_file.name
            joblib.dump(obj, tmp_path)
            self.client.upload(hdfs_path, tmp_path, overwrite=True)
            print(f"Объект сохранен в HDFS: {hdfs_path}")
        except HdfsError as e:
            print(f"Ошибка записи объекта в HDFS {hdfs_path}: {e}")
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def read_joblib(self, hdfs_path: str) -> Any:
        """Чтение объекта из HDFS через joblib."""
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                tmp_path = tmp_file.name
            self.client.download(hdfs_path, tmp_path, overwrite=True)
            return joblib.load(tmp_path)
        except Exception as e:
            print(f"Ошибка чтения объекта из HDFS {hdfs_path}: {e}")
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
