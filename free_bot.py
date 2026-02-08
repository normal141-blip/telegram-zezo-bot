import os
import asyncio
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# مؤشرات (ta)
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice


# ========= إعدادات =========
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("ضع TELEGRAM_BOT_TOKEN داخل ملف .env في نفس مجلد البوت")


def fetch_data(symbol: str) -> pd.DataFrame:
    # نحاول أولاً بيانات قصيرة (إذا السوق مفتوح)
    df = yf.download(
        tickers=symbol,
        period="5d",
        interval="5m",
        progress=False,
        threads=False
    )

    # إذا لم توجد بيانات، نجرب بيانات يومية (يعمل حتى لو السوق مغلق)
    if df.empty:
        df = yf.download(
            tickers=symbol,
            period="1mo",
            interval="1d",
            progress=False,
            threads=False
        )

    return df
    
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()


def compute_signals(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    ema9 = EMAIndicator(close=close, window=9).ema_indicator()
    ema21 = EMAIndicator(close=close, window=21).ema_indicator()
    rsi = RSIIndicator(close=close, window=14).rsi()

    vwap = VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=vol, window=14
    ).volume_weighted_average_price()

    avg_vol = vol.tail(20).mean()
    current_vol = float(vol.iloc[-1]) if len(vol) else 0.0
    volume_ratio = (current_vol / float(avg_vol)) if (avg_vol and avg_vol > 0) else 1.0

    last_close = float(close.iloc[-1])
    last_ema9 = float(ema9.iloc[-1])
    last_ema21 = float(ema21.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_vwap = float(vwap.iloc[-1])

    recent = df.tail(20)
    range_val = float((recent["High"] - recent["Low"]).mean())
    if (not range_val) or pd.isna(range_val) or range_val <= 0:
        range_val = float(df["High"].iloc[-1] - df["Low"].iloc[-1])
    if range_val <= 0:
        range_val = max(last_close * 0.002, 0.5)

    return {
        "close": last_close,
        "ema9": last_ema9,
        "ema21": last_ema21,
        "rsi": last_rsi,
        "vwap": last_vwap,
        "range": range_val,
        "volume_ratio": float(volume_ratio),
    }


def decide_recommendation(sig: dict) -> tuple[str, str, int]:
    close = sig["close"]
    ema9 = sig["ema9"]
    ema21 = sig["ema21"]
    rsi = sig["rsi"]
    vwap = sig["vwap"]
    volume_ratio = sig["volume_ratio"]

    buy_score = 0
    sell_score = 0
    reasons = []

    # EMA اتجاه (وزن أعلى)
    if ema9 > ema21:
        buy_score += 2
        reasons.append("اتجاه صاعد (EMA)")
    elif ema9 < ema21:
        sell_score += 2
        reasons.append("اتجاه هابط (EMA)")
    else:
        reasons.append("EMA متعادل")

    # VWAP
    if close > vwap:
        buy_score += 1
        reasons.append("فوق VWAP")
    else:
        sell_score += 1
        reasons.append("تحت VWAP")

    # RSI
    if 40 <= rsi <= 65:
        buy_score += 1
        reasons.append("RSI صحي")
    elif rsi >= 70:
        sell_score += 1
        reasons.append("تشبع شراء (RSI)")
    elif rsi <= 30:
        buy_score += 1
        reasons.append("تشبع بيع (RSI)")
    else:
        reasons.append("RSI طبيعي")

    # Volume Spike
    if volume_ratio >= 1.5:
        buy_score += 1
        reasons.append("حجم قوي")
    else:
        reasons.append("حجم عادي")

    max_score = max(buy_score, sell_score)
    strength = int(min(100, (max_score / 5) * 100))

    if buy_score >= 3 and buy_score > sell_score:
        return "🟢 شراء", " + ".join(reasons), strength
    if sell_score >= 3 and sell_score > buy_score:
        return "🔴 بيع", " + ".join(reasons), strength

    if buy_score > sell_score:
        return "🟡 انتظار (ميل شراء)", " + ".join(reasons), strength
    if sell_score > buy_score:
        return "🟡 انتظار (ميل بيع)", " + ".join(reasons), strength

    return "🟡 انتظار", " + ".join(reasons), strength


def build_levels(sig: dict, rec: str) -> dict:
    entry = sig["close"]
    r = sig["range"]
    if r <= 0:
        r = max(entry * 0.002, 0.5)

    if "بيع" in rec:
        sl = entry + r
        t1 = entry - r
        t2 = entry - 2 * r
        t3 = entry - 3 * r
        t4 = entry - 4 * r
    else:
        sl = entry - r
        t1 = entry + r
        t2 = entry + 2 * r
        t3 = entry + 3 * r
        t4 = entry + 4 * r

    return {"entry": entry, "sl": sl, "t1": t1, "t2": t2, "t3": t3, "t4": t4}


def analyze(symbol: str) -> str:
    df = fetch_data(symbol)
    if df.empty:
        return "❌ لا توجد بيانات (السوق مغلق أو الرمز خطأ)"

    sig = compute_signals(df)
    rec, why, strength = decide_recommendation(sig)
    lv = build_levels(sig, rec)

    return (
        f"📊 {symbol}\n"
        f"🕌 الشرعية: (مسؤوليتك أنت)\n"
        f"📌 التوصية: {rec} (مسؤوليتك أنت)\n"
        f"💪 قوة الإشارة: {strength}%\n"
        f"🧠 السبب: {why}\n"
        f"📈 RSI: {sig['rsi']:.1f} | EMA9: {sig['ema9']:.2f} | EMA21: {sig['ema21']:.2f} | VWAP: {sig['vwap']:.2f}\n"
        f"💰 دخول: {lv['entry']:.2f}\n"
        f"🛑 وقف خسارة: {lv['sl']:.2f}\n"
        f"🎯 هدف 1: {lv['t1']:.2f}\n"
        f"🎯 هدف 2: {lv['t2']:.2f}\n"
        f"🎯 هدف 3: {lv['t3']:.2f}\n"
        f"🎯 هدف 4: {lv['t4']:.2f}"
    )


# ===== Telegram =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 مرحبًا بك في بوت زيزو لتحليل الأسهم\n\n"
        "📈 هذا البوت يقوم بتحليل الأسهم باستخدام الذكاء الاصطناعي.\n\n"
        "✉️ فقط أرسل رمز السهم مثل:\n"
        "AAPL\n"
        "TSLA\n"
        "NVDA\n\n"
        "📊 سيقوم البوت بتحليل السهم وإظهار الفرص المتاحة.\n"
        "⚠️ القرار الاستثماري يعود لك."
    )


async def handle_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    text = (update.message.text or "").strip()
    if not text or text.startswith("/"):
        return

    symbol = text.upper()
    await update.message.reply_text(f"⏳ جاري التحليل: {symbol}")

    try:
        result = await asyncio.to_thread(analyze, symbol)
        await update.message.reply_text(result)
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ: {type(e).__name__}")


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_symbol))

    print("BOT IS RUNNING...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False
    )


if __name__ == "__main__":
    main()

