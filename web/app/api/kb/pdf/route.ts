export const runtime = "nodejs";

export async function GET() {
  return new Response(null, { status: 404 });
}
