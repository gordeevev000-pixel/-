"""Все параметры метода в одном месте. Выбраны заранее, а не подобраны на тесте."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- рынок
    market_tz: str = "UTC"          # пояс, в котором считать время суток и выходные
    day_start_hour: int = 0         # начало торгового дня в market_tz (FX: 17 по America/New_York)
    max_fill: int = 3               # дыры до N минут = «минуты без сделок», длиннее — разрыв ряда
    # --- отпечаток
    atr_n: int = 60                 # ATR минутных баров, в котором меряется всё
    atr_short: int = 15
    atr_long: int = 7200            # «средняя» волатильность: ~5 суток торговли 24/7
    atr_floor: float = 0.3          # ATR не ниже 0.3 x средней (защита от мёртвых минут)
    lags: tuple = tuple(range(1, 16)) + tuple(range(18, 61, 3))  # 15 точек по 1 мин + 15 по 3 мин = 60 мин
    eff_n: tuple = (15, 30, 60)     # окна эффективности движения
    # --- исход
    horizon: int = 15
    barrier_atr: float = 1.0        # барьер «+1 ATR раньше −1 ATR»
    # --- поиск
    k: int = 300                    # число аналогов
    k_candidates: int = 4000        # сколько брать из индекса до прореживания по эпизодам
    episode_gap: int = 30           # не больше одного аналога на окно 30 минут
    # веса блоков отпечатка (доля в квадрате расстояния ~ w^2); заданы априори
    block_weights: dict = field(default_factory=lambda: {
        "path": 1.0, "vol": 0.6, "tod": 0.5, "dayhl": 0.5, "eff": 0.5})
    # --- живое сужение
    live_sigma: float = 0.5         # масштаб ошибки пути (в ATR) при перевзвешивании
    alive_rmse: float = 0.5         # аналог «жив», если RMSE его первых k минут <= этого порога
