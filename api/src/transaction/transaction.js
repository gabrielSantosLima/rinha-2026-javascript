import * as z from "zod";

export class Transaction {
  constructor(data) {
    this.content = this.validateTransactionData(data);
  }

  validateTransactionData(data) {
    const transactionSchema = z.object({
      id: z.string(),
      transaction: z.object({
        amount: z.number(),
        installments: z.number().int(),
        requested_at: z.string().refine((date) => !isNaN(Date.parse(date)), {
          message: "Invalid date format for requested_at",
        }),
      }),
      customer: z.object({
        avg_amount: z.number(),
        tx_count_24h: z.number().int(),
        known_merchants: z.array(z.string()),
      }),
      merchant: z.object({
        id: z.string(),
        mcc: z.string(),
        avg_amount: z.number().positive(),
      }),
      terminal: z.object({
        is_online: z.boolean(),
        card_present: z.boolean(),
        km_from_home: z.number(),
      }),
      last_transaction: z
        .object({
          timestamp: z.string().refine((date) => !isNaN(Date.parse(date)), {
            message: "Invalid date format for last_transaction.timestamp",
          }),
          km_from_current: z.number(),
        })
        .nullable(),
    });
    return transactionSchema.parse(data);
  }
}
