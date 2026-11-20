import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@mui/material";
import { noop } from "es-toolkit";
import { type HTMLProps, useMemo } from "react";

import { useI18n } from "../../context/useI18n";

import { type ColumnContext, prepareColumns, trendColumns } from "./columns";

export interface StatsItem {
  games: number;
  move: string;
  points: number;
  stats: Record<number, MoveStats>;
  years: number[];
}

interface MoveStats {
  count: number;
  points: number;
}

interface PositionMovesProperties extends HTMLProps<HTMLDivElement> {
  readonly doMove?: (move: string) => void;
  readonly stats: StatsItem[];
}

const PositionMoves: React.FC<PositionMovesProperties> = ({
  doMove = noop,
  stats,
  ...properties
}) => {
  const { t } = useI18n();

  const currentYear = new Date().getFullYear();

  const context = useMemo<ColumnContext>(
    () => ({ currentYear }),
    [currentYear],
  );

  const columns = trendColumns;

  const prepared = useMemo(
    () => prepareColumns(columns, stats, context),
    [columns, stats, context],
  );

  if (stats.length === 0) {
    return null;
  }

  return (
    <div {...properties}>
      <Table className="colorful-table" id="stats">
        <TableHead>
          <TableRow>
            <TableCell>{t("move")}</TableCell>
            <TableCell>{t("no_games")}</TableCell>
            <TableCell>%</TableCell>
            <TableCell>{t("latest")}</TableCell>

            {columns.map((column) => (
              <TableCell key={column.id}>{column.label}</TableCell>
            ))}
          </TableRow>
        </TableHead>

        <TableBody>
          {stats.map((item, index) => (
            <TableRow key={item.move} onClick={() => doMove(item.move)}>
              <TableCell>{item.move}</TableCell>
              <TableCell>{item.games}</TableCell>
              <TableCell>
                {((item.points / item.games) * 100).toFixed(2)}
              </TableCell>
              <TableCell>
                {item.years.length > 0 ? Math.max(...item.years) : "—"}
              </TableCell>

              {columns.map((column) => {
                const key = column.transformKey ?? column.id;

                return (
                  <TableCell key={column.id}>
                    {column.calculate(item, index, prepared.get(key), context)}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default PositionMoves;
