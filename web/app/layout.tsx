import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "女巫自习室 · 霍格沃茨图书馆",
  description: "女巫自习室 · 霍格沃茨图书馆"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className="shell">
          <header className="topbar">
            <div className="brand">
              <div className="brandTitle">女巫自习室 · 霍格沃茨图书馆</div>
              <div className="brandSub">把问题交给羽毛笔</div>
            </div>
          </header>
          <main className="container">{children}</main>
        </div>
      </body>
    </html>
  );
}
