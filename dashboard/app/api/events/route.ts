

import { NextRequest, NextResponse } from "next/server";
import { askExecutive } from "@/lib/agent";

export async function POST(req: NextRequest) {
  const { question, region } = await req.json();
  const result = await askExecutive(question, { region });
  return NextResponse.json(result); // {summary, data[], sources?}
}