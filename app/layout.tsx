import type { Metadata } from "next";
import Analytics from "./Analytics";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ||
      "https://wgblearn-coder.github.io/",
  ),
  title: {
    default: "wuguobin · 独立开发者与 AI 实践者",
    template: "%s · wuguobin",
  },
  description:
    "记录 AI 产品、全栈开发、多智能体与个人知识系统的实践和思考。",
  openGraph: {
    type: "website",
    locale: "zh_CN",
    siteName: "wuguobin · 日新知见",
    title: "wuguobin · 独立开发者与 AI 实践者",
    description:
      "记录 AI 产品、全栈开发、多智能体与个人知识系统的实践和思考。",
  },
  twitter: {
    card: "summary_large_image",
    title: "wuguobin · 独立开发者与 AI 实践者",
    description:
      "记录 AI 产品、全栈开发、多智能体与个人知识系统的实践和思考。",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <head>
        <Analytics />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
