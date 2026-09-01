import type {
  Metadata,
} from "next";

import "./globals.css";


export const metadata: Metadata = {
  title:
    "DataLens · Local Analytical Intelligence",

  description:
    "Local-first analytical intelligence for deterministic analysis, grounded AI and defensible decisions.",
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