import type {
  Metadata,
} from "next";

import "./globals.css";


export const metadata: Metadata = {
  title:
    "DataLens · Local analytical workspace",

  description:
    "Local-first data analysis, statistics, visualizations and explainable dashboards.",
};


export default function RootLayout({
  children,
}: Readonly<{
  children:
    React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body>
        {
          children
        }
      </body>
    </html>
  );
}