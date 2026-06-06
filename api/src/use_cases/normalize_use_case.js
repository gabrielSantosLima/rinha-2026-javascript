import { Transaction } from "../transaction/transaction.js";

const MCC_RISK = {
  5411: 0.15,
  5812: 0.3,
  5912: 0.2,
  5944: 0.45,
  7801: 0.8,
  7802: 0.75,
  7995: 0.85,
  4511: 0.35,
  5311: 0.25,
  5999: 0.5,
};

const CONFIG = {
  max_amount: 10000,
  max_installments: 12,
  amount_vs_avg_ratio: 10,
  max_minutes: 1440,
  max_km: 1000,
  max_tx_count_24h: 20,
  max_merchant_avg_amount: 10000,
};

export class NormalizeUseCase {
  clamp(value) {
    return Math.min(1, Math.max(0, value));
  }

  round(value) {
    return value === -1 ? -1 : Number(value.toFixed(4));
  }

  execute(data) {
    const tx = new Transaction(data).content;

    const requestedAt = new Date(tx.transaction.requested_at);

    const amount = this.clamp(tx.transaction.amount / CONFIG.max_amount);

    const installments = this.clamp(
      tx.transaction.installments / CONFIG.max_installments,
    );

    const customerAvg = tx.customer.avg_amount > 0 ? tx.customer.avg_amount : 1;

    const amountVsAvg = this.clamp(
      tx.transaction.amount / customerAvg / CONFIG.amount_vs_avg_ratio,
    );

    const hour = requestedAt.getUTCHours() / 23;

    // JS: Sun=0 ... Sat=6
    // Spec: Mon=0 ... Sun=6
    const dayOfWeek = ((requestedAt.getUTCDay() + 6) % 7) / 6;

    let minutesSinceLast = -1;
    let kmFromLast = -1;

    if (tx.last_transaction) {
      const lastTimestamp = new Date(tx.last_transaction.timestamp);

      const minutes = (requestedAt.getTime() - lastTimestamp.getTime()) / 60000;

      minutesSinceLast = this.clamp(minutes / CONFIG.max_minutes);

      kmFromLast = this.clamp(
        tx.last_transaction.km_from_current / CONFIG.max_km,
      );
    }

    const kmFromHome = this.clamp(tx.terminal.km_from_home / CONFIG.max_km);

    const txCount = this.clamp(
      tx.customer.tx_count_24h / CONFIG.max_tx_count_24h,
    );

    const isOnline = tx.terminal.is_online ? 1 : 0;

    const cardPresent = tx.terminal.card_present ? 1 : 0;

    // Spec: 1 = unknown merchant
    const unknownMerchant = tx.customer.known_merchants.includes(tx.merchant.id)
      ? 0
      : 1;

    const mccRisk = MCC_RISK[tx.merchant.mcc] ?? 0.5;

    const merchantAvg = this.clamp(
      tx.merchant.avg_amount / CONFIG.max_merchant_avg_amount,
    );

    return [
      this.round(amount),
      this.round(installments),
      this.round(amountVsAvg),
      this.round(hour),
      this.round(dayOfWeek),
      minutesSinceLast === -1 ? -1 : this.round(minutesSinceLast),
      kmFromLast === -1 ? -1 : this.round(kmFromLast),
      this.round(kmFromHome),
      this.round(txCount),
      isOnline,
      cardPresent,
      unknownMerchant,
      this.round(mccRisk),
      this.round(merchantAvg),
    ];
  }
}
