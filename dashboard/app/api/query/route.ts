// app/api/query/route.ts
/**
 * API Route for Natural Language Risk Queries
 * Handles executive questions and returns structured responses
 */

import { NextRequest, NextResponse } from "next/server";
import { askExecutive, type ExecutiveContext } from "@/lib/agent";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * POST /api/query
 * Body: { question: string, context?: ExecutiveContext }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { question, context } = body;

    // Validate input
    if (!question || typeof question !== "string") {
      return NextResponse.json(
        { error: "Question is required and must be a string" },
        { status: 400 }
      );
    }

    if (question.length > 500) {
      return NextResponse.json(
        { error: "Question is too long (max 500 characters)" },
        { status: 400 }
      );
    }

    // Optional: Add authentication/authorization here
    // const session = await getServerSession(authOptions);
    // if (!session) {
    //   return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    // }

    // Execute query
    const execContext: ExecutiveContext = {
      region: context?.region,
      days: context?.days,
      userRole: context?.userRole || "executive",
    };

    console.log(`[Query API] Processing: "${question}"`, execContext);

    const result = await askExecutive(question, execContext);

    // Log for analytics
    console.log(
      `[Query API] Success: ${result.data.length} risks, ${
        result.recommendations?.length || 0
      } recommendations`
    );

    return NextResponse.json({
      success: true,
      result,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("[Query API] Error:", error);

    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";

    return NextResponse.json(
      {
        success: false,
        error: errorMessage,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/query?q=<question>&region=<region>&days=<days>
 * Simple query parameter based endpoint
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const question = searchParams.get("q");
    const region = searchParams.get("region");
    const days = searchParams.get("days");

    if (!question) {
      return NextResponse.json(
        { error: "Query parameter 'q' is required" },
        { status: 400 }
      );
    }

    const execContext: ExecutiveContext = {
      region: region || undefined,
      days: days ? parseInt(days) : undefined,
      userRole: "executive",
    };

    const result = await askExecutive(question, execContext);

    return NextResponse.json({
      success: true,
      result,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("[Query API] Error:", error);

    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";

    return NextResponse.json(
      {
        success: false,
        error: errorMessage,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
