import { NextRequest, NextResponse } from "next/server";

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api).*)"]
};
