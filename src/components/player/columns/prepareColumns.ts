import type { StatsItem } from "../PositionsMoves";

import type { ColumnContext, ColumnDefinition } from "./types";

export type PreparedColumns = Map<string, unknown>;

export function prepareColumns(
  columns: readonly ColumnDefinition[],
  rows: readonly StatsItem[],
  context: ColumnContext,
): PreparedColumns {
  const prepared: PreparedColumns = new Map();

  for (const column of columns) {
    if (!column.transform) {
      continue;
    }

    const key = column.transformKey ?? column.id;

    if (!prepared.has(key)) {
      prepared.set(key, column.transform(rows, context));
    }
  }

  return prepared;
}
