import type { ReactNode } from "react";

import type { StatsItem } from "../PositionsMoves";

export interface ColumnContext {
  currentYear: number;
}

export interface ColumnDefinition {
  calculate: (
    row: StatsItem,
    index: number,
    transformed: unknown,
    context: ColumnContext,
  ) => ReactNode;
  id: string;
  label: string;

  transform?: (rows: readonly StatsItem[], context: ColumnContext) => unknown;
  transformKey?: string;
  visibleByDefault?: boolean;
}
