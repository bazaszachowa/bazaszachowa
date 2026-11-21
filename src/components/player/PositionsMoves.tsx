import {
  Checkbox,
  FormControlLabel,
  FormGroup,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@mui/material";
import { noop } from "es-toolkit";
import Cookies from "js-cookie";
import { type HTMLProps, useEffect, useMemo, useState } from "react";

import type { GameData } from "../../ChessEditor";
import { useI18n } from "../../context/useI18n";
import { usePlayerPredictions } from "../../hooks/usePlayerPredictions";

import { type ColumnContext, prepareColumns, trendColumns } from "./columns";
import { createAIColumns } from "./columns/trendColumns";

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

const VISIBLE_COLUMNS_COOKIE = "position-moves-visible-columns";

const DEFAULT_VISIBLE_COLUMNS = ["alfa", "beta", "ai-game-share"];

const getVisibleColumns = (): string[] => {
  const cookie = Cookies.get(VISIBLE_COLUMNS_COOKIE);

  if (cookie === undefined) {
    return DEFAULT_VISIBLE_COLUMNS;
  }

  try {
    const parsed: unknown = JSON.parse(cookie);

    if (
      Array.isArray(parsed) &&
      parsed.every((item): item is string => typeof item === "string")
    ) {
      return parsed;
    }
  } catch {}

  return DEFAULT_VISIBLE_COLUMNS;
};
interface PositionMovesProperties extends HTMLProps<HTMLDivElement> {
  readonly doMove?: (move: string) => void;
  readonly games: readonly GameData[];
  readonly stats: StatsItem[];
}

const PositionMoves: React.FC<PositionMovesProperties> = ({
  doMove = noop,
  games,
  stats,
  ...properties
}) => {
  const { t } = useI18n();

  const currentYear = new Date().getFullYear();

  const { error, loading, predictions } = usePlayerPredictions(games, stats);

  const [visibleColumnIds, setVisibleColumnIds] =
    useState<string[]>(getVisibleColumns);

  const context = useMemo<ColumnContext>(
    () => ({ currentYear, games }),
    [currentYear, games],
  );

  const columns = useMemo(
    () => [...trendColumns, ...createAIColumns(predictions)],
    [predictions],
  );

  const visibleColumns = useMemo(
    () => columns.filter((column) => visibleColumnIds.includes(column.id)),
    [columns, visibleColumnIds],
  );

  const prepared = useMemo(
    () => prepareColumns(visibleColumns, stats, context),
    [visibleColumns, stats, context],
  );

  const toggleColumn = (id: string, checked: boolean) => {
    setVisibleColumnIds((current) =>
      checked
        ? [...current, id]
        : current.filter((columnId) => columnId !== id),
    );
  };

  useEffect(() => {
    Cookies.set(VISIBLE_COLUMNS_COOKIE, JSON.stringify(visibleColumnIds), {
      expires: 365,
      sameSite: "Lax",
    });
  }, [visibleColumnIds]);
  if (stats.length === 0) {
    return null;
  }

  return (
    <div {...properties}>
      <FormGroup row>
        {columns.map((column) => (
          <FormControlLabel
            control={
              <Checkbox
                checked={visibleColumnIds.includes(column.id)}
                onChange={(event) =>
                  toggleColumn(column.id, event.target.checked)
                }
                size="small"
              />
            }
            key={column.id}
            label={column.label}
          />
        ))}
      </FormGroup>

      <Table className="colorful-table" id="stats">
        <TableHead>
          <TableRow>
            <TableCell>{t("move")}</TableCell>
            <TableCell>{t("no_games")}</TableCell>
            <TableCell>%</TableCell>
            <TableCell>{t("latest")}</TableCell>

            {visibleColumns.map((column) => (
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

              {visibleColumns.map((column) => {
                const isAIColumn = column.id.startsWith("ai-");

                return (
                  <TableCell key={column.id}>
                    {isAIColumn && (loading || error)
                      ? "—"
                      : column.calculate(
                          item,
                          index,
                          prepared.get(column.transformKey ?? column.id),
                          context,
                        )}
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
