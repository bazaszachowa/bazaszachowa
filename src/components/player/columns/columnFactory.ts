import type { ReactNode } from "react";

import type { StatsItem } from "../PositionsMoves";

import type { ColumnContext, ColumnDefinition } from "./types";

interface ColumnOptions<TTransformed, TValue> {
  calculate: (
    row: StatsItem,
    index: number,
    transformed: TTransformed,
    context: ColumnContext,
  ) => TValue;
  id: string;
  label: string;
  render: (value: TValue, row: StatsItem, index: number) => ReactNode;

  transform?: (
    rows: readonly StatsItem[],
    context: ColumnContext,
  ) => TTransformed;

  transformKey?: string;

  visibleByDefault?: boolean;
}

export function defineColumn<TTransformed, TValue>(
  options: ColumnOptions<TTransformed, TValue>,
): ColumnDefinition {
  return {
    calculate: (row, index, transformed, context) => {
      const value = options.calculate(
        row,
        index,
        transformed as TTransformed,
        context,
      );

      return options.render(value, row, index);
    },
    id: options.id,
    label: options.label,
    transform: options.transform,
    transformKey: options.transformKey,

    visibleByDefault: options.visibleByDefault,
  };
}
