import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
} from "@mui/material";
import { useMemo } from "react";

import { useI18n } from "../context/useI18n";

import Markdown from "./app/Markdown";

const mainBoxStyles = {
  "& .MuiAccordion-root": {
    boxSizing: "border-box",
    maxWidth: "100%",
    minWidth: 0,
    width: "100%",
  },
  "& .MuiAccordionDetails-root": {
    boxSizing: "border-box",
    maxWidth: "100%",
    minWidth: 0,
    overflowX: "auto",
  },
  boxSizing: "border-box",
  containerType: "inline-size",
  display: "block",
  maxWidth: "100%",
  minWidth: 0,
  overflow: "clip",
  width: "100%",
} as const;
const innerBoxStyle = {
  "& img": { maxWidth: "100%" },
  "& pre": {
    maxWidth: "100%",
    overflowX: "auto",
  },
  "& table": {
    display: "block",
    maxWidth: "100%",
    overflowX: "auto",
  },
  maxWidth: "100%",
  minWidth: 0,
  overflowWrap: "anywhere",
};
const detailsStyle = {
  boxSizing: "border-box",
  maxWidth: "100%",
  minWidth: 0,
  overflowX: "auto",
};

const TrendFunctionExplanation = () => {
  const { t } = useI18n();

  const sections = useMemo(
    () => [
      { section: "alpha", title: `${t("math.function")} α` },
      { section: "beta", title: `${t("math.function")} β` },
      { section: "absolute", title: `${t("math.ai_model")} M1` },
      { section: "percentage", title: `${t("math.ai_model")} M2` },
    ],
    [t],
  );

  return (
    <Box sx={mainBoxStyles}>
      {sections.map(({ section, title }) => (
        <Accordion disableGutters key={section}>
          <AccordionSummary>{title}</AccordionSummary>
          <AccordionDetails sx={detailsStyle}>
            <Box sx={innerBoxStyle}>
              <Markdown screen="math" section={section} />
            </Box>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default TrendFunctionExplanation;
