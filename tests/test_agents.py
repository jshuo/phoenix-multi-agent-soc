# tests/test_agents.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain_community.vectorstores import FAISS
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.agents import Tool

import json, re
from collections import Counter
from pathlib import Path

def build_memory():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Enhanced initial memory with security context
    initial_context = [
        "AutoGPT Fab Security Assistant - Specialized in detailed threat analysis and comprehensive mitigations",
        "Mission: Provide actionable, step-by-step security recommendations with business justification",
        "Focus Areas: Authentication security, firmware integrity, data loss prevention, process parameter control",
        "Output Requirements: Technical implementation details, cost estimates, timelines, risk assessments"
    ]
    vs = FAISS.from_texts(initial_context, embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": 6})

def _load_fab_logs(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"fab logs not found at {path}")
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _create_detailed_mitigations(anomaly_type: str, count: int, examples: list) -> dict:
    """Create comprehensive mitigation details for each anomaly type"""
    
    mitigation_templates = {
        "login_failures": {
            "threat_description": "Brute force and credential stuffing attacks targeting authentication systems",
            "business_risk": "Unauthorized access leading to IP theft, operational disruption, regulatory violations",
            "financial_impact": "$500K - $2M potential loss per successful breach",
            "technical_solution": "Multi-factor authentication with FIDO2/WebAuthn hardware tokens",
            "implementation_steps": [
                "1. Deploy FIDO2-compatible authentication infrastructure",
                "2. Configure account lockout policies (3 failed attempts = 15min lockout)",
                "3. Implement real-time login monitoring with SIEM integration", 
                "4. Set up automated alerting for suspicious login patterns",
                "5. Deploy privileged access management (PAM) for admin accounts",
                "6. Configure network segmentation for authentication servers"
            ],
            "technologies_required": [
                "FIDO2/WebAuthn tokens ($50-100 per user)",
                "Identity Provider (Okta/Azure AD: $2-8 per user/month)",
                "SIEM platform (Splunk/QRadar: $150-300 per GB/day)",
                "PAM solution (CyberArk/BeyondTrust: $100-200 per user/year)"
            ],
            "timeline": "4-6 weeks implementation",
            "personnel_required": "2 security engineers, 1 network admin (80 hours total)",
            "success_metrics": "Reduce successful brute force by 99%, achieve <1% false positive rate",
            "compliance_benefits": "Meets SOX, NIST, ISO 27001 authentication requirements"
        },
        
        "unsigned_firmware": {
            "threat_description": "Supply chain attacks and malicious firmware injection",
            "business_risk": "Complete system compromise, industrial espionage, production sabotage",
            "financial_impact": "$5M - $50M potential loss from compromised production systems",
            "technical_solution": "HSM-based firmware signing with cryptographic verification",
            "implementation_steps": [
                "1. Deploy dedicated Hardware Security Module (HSM) infrastructure",
                "2. Establish secure firmware signing workflow and approval process",
                "3. Create cryptographically signed firmware allowlist database",
                "4. Implement boot-time signature verification on all systems",
                "5. Deploy firmware integrity monitoring agents",
                "6. Configure automated quarantine for unsigned firmware detection",
                "7. Establish secure firmware distribution channels"
            ],
            "technologies_required": [
                "Hardware Security Module (Thales/Utimaco: $15K-50K)",
                "Code signing certificates and PKI infrastructure ($5K-15K annual)",
                "Firmware verification agents (custom development: $50K-100K)",
                "Secure firmware repository (custom/commercial: $20K-40K)"
            ],
            "timeline": "8-12 weeks implementation",
            "personnel_required": "3 security engineers, 2 firmware developers, 1 crypto specialist (200+ hours)",
            "success_metrics": "100% firmware signature verification, zero unsigned firmware deployments",
            "compliance_benefits": "Meets NIST Cybersecurity Framework, IEC 62443 industrial security standards"
        },
        
        "data_egress": {
            "threat_description": "Intellectual property theft and sensitive data exfiltration",
            "business_risk": "Loss of competitive advantage, regulatory fines, customer data exposure",
            "financial_impact": "$1M - $10M from IP theft, $2.9M average data breach cost",
            "technical_solution": "Data Loss Prevention (DLP) with network traffic analysis",
            "implementation_steps": [
                "1. Deploy network DLP sensors on all egress points",
                "2. Configure data classification and labeling policies",
                "3. Implement egress rate limiting (100MB/hour baseline)",
                "4. Set up content inspection for sensitive data patterns",
                "5. Deploy Cloud Access Security Broker (CASB) for cloud services",
                "6. Configure real-time alerting for large data transfers",
                "7. Implement user behavior analytics (UBA) for anomaly detection"
            ],
            "technologies_required": [
                "Network DLP solution (Symantec/Forcepoint: $100K-300K)",
                "CASB platform (Netskope/McAfee: $10-25 per user/month)",
                "Network monitoring tools (SolarWinds/PRTG: $20K-50K)",
                "UBA platform (Exabeam/Splunk: $50K-150K annual)"
            ],
            "timeline": "6-8 weeks implementation",
            "personnel_required": "2 network security engineers, 1 data analyst (120 hours total)",
            "success_metrics": "Block 95% of unauthorized data transfers, detect exfiltration within 5 minutes",
            "compliance_benefits": "Supports GDPR, CCPA, SOX data protection requirements"
        },
        
        "parameter_drifts": {
            "threat_description": "Unauthorized process modifications and industrial sabotage",
            "business_risk": "Product quality issues, safety incidents, production downtime",
            "financial_impact": "$100K - $5M from quality issues and production disruption",
            "technical_solution": "Digital signature verification for parameter changes with dual-control",
            "implementation_steps": [
                "1. Implement digital signature system for all parameter changes",
                "2. Configure dual-operator approval workflow (two-person integrity)",
                "3. Deploy parameter change monitoring and alerting system",
                "4. Create comprehensive audit trail for all modifications",
                "5. Implement parameter baseline validation and drift detection",
                "6. Set up automated rollback capabilities for unauthorized changes",
                "7. Configure emergency override procedures with enhanced logging"
            ],
            "technologies_required": [
                "Digital signature infrastructure (integrated with existing PKI: $10K-25K)",
                "Workflow management system (ServiceNow/Jira: $10-50 per user/month)",
                "Parameter monitoring agents (custom development: $30K-60K)",
                "Audit logging platform (Splunk add-on: $10K-20K)"
            ],
            "timeline": "3-5 weeks implementation", 
            "personnel_required": "2 process engineers, 1 security engineer (60 hours total)",
            "success_metrics": "100% parameter change authorization, detect drift within 1 minute",
            "compliance_benefits": "Supports FDA 21 CFR Part 11, ISO 9001 quality management"
        }
    }
    
    return mitigation_templates.get(anomaly_type, {})

def _summarize_anomalies(rows: list[dict]) -> str:
    """Enhanced analysis with detailed mitigations and business context"""
    msgs = [r["msg"] for r in rows]
    fails = sum("login failure" in m.lower() for m in msgs)
    unsigned_fw = [m for m in msgs if re.search(r"(signature:\s*MISSING|verify signature FAILED)", m, re.I)]
    exfil = [m for m in msgs if re.search(r"(SCP transfer|GB to )", m, re.I)]
    param_drifts = [m for m in msgs if re.search(r"(baseline\s*\d+)", m, re.I)]

    hosts = Counter(r["host"] for r in rows)
    top_host, top_count = hosts.most_common(1)[0]
    
    # Calculate risk scores
    risk_score = (fails * 2) + (len(unsigned_fw) * 10) + (len(exfil) * 8) + (len(param_drifts) * 3)
    risk_level = "CRITICAL" if risk_score >= 20 else "HIGH" if risk_score >= 10 else "MEDIUM"
    
    # Generate detailed mitigations
    detailed_mitigations = {}
    
    if fails > 0:
        detailed_mitigations["authentication_security"] = _create_detailed_mitigations("login_failures", fails, [])
        detailed_mitigations["authentication_security"]["anomaly_count"] = fails
        
    if len(unsigned_fw) > 0:
        detailed_mitigations["firmware_integrity"] = _create_detailed_mitigations("unsigned_firmware", len(unsigned_fw), unsigned_fw[:3])
        detailed_mitigations["firmware_integrity"]["anomaly_count"] = len(unsigned_fw)
        
    if len(exfil) > 0:
        detailed_mitigations["data_protection"] = _create_detailed_mitigations("data_egress", len(exfil), exfil[:3])
        detailed_mitigations["data_protection"]["anomaly_count"] = len(exfil)
        
    if len(param_drifts) > 0:
        detailed_mitigations["process_integrity"] = _create_detailed_mitigations("parameter_drifts", len(param_drifts), param_drifts[:3])
        detailed_mitigations["process_integrity"]["anomaly_count"] = len(param_drifts)

    # Executive summary
    executive_summary = {
        "threat_assessment": {
            "overall_risk_level": risk_level,
            "risk_score": f"{risk_score}/100",
            "total_anomalies": fails + len(unsigned_fw) + len(exfil) + len(param_drifts),
            "affected_systems": len(hosts),
            "primary_threat_vector": top_host,
            "immediate_action_required": len(unsigned_fw) > 0 or fails > 5 or len(exfil) > 0
        },
        "business_impact_summary": {
            "estimated_total_risk_exposure": f"${(fails * 1000000) + (len(unsigned_fw) * 20000000) + (len(exfil) * 5000000) + (len(param_drifts) * 2000000):,}",
            "regulatory_compliance_risk": "HIGH" if (fails > 0 or len(exfil) > 0) else "MEDIUM",
            "operational_disruption_risk": "CRITICAL" if len(unsigned_fw) > 0 else "HIGH",
            "competitive_advantage_risk": "HIGH" if len(exfil) > 0 else "LOW"
        },
        "implementation_overview": {
            "total_estimated_cost": f"${sum([150000 if 'authentication_security' in detailed_mitigations else 0, 300000 if 'firmware_integrity' in detailed_mitigations else 0, 200000 if 'data_protection' in detailed_mitigations else 0, 75000 if 'process_integrity' in detailed_mitigations else 0]):,}",
            "estimated_timeline": "8-12 weeks for complete implementation",
            "personnel_required": "6-8 security engineers, 2-3 specialists",
            "expected_risk_reduction": "85-95% reduction in identified threat vectors"
        }
    }

    # Comprehensive response format
    summary = f"""
COMPREHENSIVE FAB SECURITY THREAT ANALYSIS
==========================================

EXECUTIVE SUMMARY:
Analyzed {len(rows)} security events across {len(hosts)} systems. Identified {risk_level} risk environment requiring immediate attention.

CRITICAL FINDINGS:
• {fails} authentication failures detected (primary attack vector)
• {len(unsigned_fw)} unsigned firmware events (supply chain risk)
• {len(exfil)} large data egress events (exfiltration risk)
• {len(param_drifts)} process parameter anomalies (integrity risk)

RISK ASSESSMENT: {risk_level} ({risk_score}/100)
PRIMARY THREAT SOURCE: {top_host} ({top_count} events)
ESTIMATED FINANCIAL EXPOSURE: {executive_summary['business_impact_summary']['estimated_total_risk_exposure']}
RECOMMENDED INVESTMENT: {executive_summary['implementation_overview']['total_estimated_cost']}
EXPECTED TIMELINE: {executive_summary['implementation_overview']['estimated_timeline']}

IMMEDIATE ACTIONS REQUIRED:
{'YES - Critical threats detected requiring emergency response' if executive_summary['threat_assessment']['immediate_action_required'] else 'NO - Standard implementation timeline acceptable'}
"""

    # Combine all data
    full_analysis = {
        "executive_summary": executive_summary,
        "threat_details": {
            "authentication_failures": fails,
            "firmware_integrity_violations": len(unsigned_fw),
            "data_exfiltration_indicators": len(exfil),
            "process_parameter_anomalies": len(param_drifts)
        },
        "affected_infrastructure": {
            "host_distribution": dict(hosts.most_common()),
            "sample_events": {
                "unsigned_firmware": unsigned_fw[:2],
                "data_egress": exfil[:2],
                "parameter_drifts": param_drifts[:2]
            }
        },
        "detailed_mitigation_strategies": detailed_mitigations
    }

    return summary + "\n\nDETAILED ANALYSIS AND MITIGATIONS:\n" + json.dumps(full_analysis, indent=2)

def analyze_fab_logs_tool(tool_input: str) -> str:
    """
    Enhanced fab security analysis tool with comprehensive mitigation strategies.
    Returns detailed threat assessment, business impact analysis, and step-by-step implementation plans.
    """
    path = tool_input.strip() or "tests/data/fab_logs.jsonl"
    try:
        rows = _load_fab_logs(path)
        return _summarize_anomalies(rows)
    except Exception as e:
        return f"ERROR: Cannot analyze fab logs at {path}: {str(e)}"

def build_tools():
    ddg = DuckDuckGoSearchRun()
    return [
        Tool(
            name="analyze_fab_logs",
            func=analyze_fab_logs_tool,
            description="Comprehensive fab security analysis with detailed mitigation strategies, cost estimates, and implementation timelines. Returns executive-ready security assessment."
        ),
        Tool(
            name="web-search", 
            func=ddg.run,
            description="Search for current security best practices, vendor solutions, and threat intelligence."
        ),
    ]

def build_llm():
    return ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,  # Slightly higher for more detailed responses
        max_tokens=3000   # Increased for comprehensive output
    )

def main():
    llm = build_llm()
    tools = build_tools()
    memory = build_memory()

    agent = AutoGPT.from_llm_and_tools(
        ai_name="FabSecBuddy-Pro",
        ai_role=(
            "You are an expert fab security consultant specializing in comprehensive threat analysis "
            "and detailed mitigation strategies. Your expertise includes OT/IT security, risk assessment, "
            "business impact analysis, and practical implementation planning. You provide executive-level "
            "security recommendations with specific technical details, cost estimates, timelines, and "
            "business justification. Your responses are thorough, actionable, and aligned with industry "
            "best practices including zero-trust architecture and defense-in-depth strategies."
        ),
        tools=tools,
        llm=llm,
        memory=memory,
    )

    # Enhanced goals for comprehensive analysis
    goals = [
        "Perform comprehensive fab security analysis using analyze_fab_logs tool on the default log file",
        "Extract and explain the detailed mitigation strategies for each identified threat category",
        "Provide executive summary with risk assessment, business impact, and investment recommendations",
        "Search for latest security best practices to validate and enhance the mitigation recommendations",
        "Compile final comprehensive security report with specific implementation guidance and timelines"
    ]

    print("=" * 80)
    print("COMPREHENSIVE FAB SECURITY ANALYSIS")
    print("=" * 80)
    print(f"Agent: {agent.ai_name}")
    print(f"Mission: Detailed threat analysis and comprehensive mitigation planning")
    print(f"Max Iterations: 4")
    print("\nAnalysis Goals:")
    for i, goal in enumerate(goals, 1):
        print(f"  {i}. {goal}")
    
    print("\n" + "=" * 80)
    print("EXECUTING COMPREHENSIVE SECURITY ANALYSIS...")
    print("=" * 80)
    
    try:
        result = agent.run(goals)
        
        print("\n" + "=" * 80)
        print("FINAL COMPREHENSIVE SECURITY ASSESSMENT")
        print("=" * 80)
        print(result)
        
        # Save results to file
        with open("comprehensive_fab_security_analysis.md", "w") as f:
            f.write(f"# Comprehensive Fab Security Analysis\n\n")
            f.write(f"Generated by: {agent.ai_name}\n\n")
            f.write(f"## Analysis Goals\n")
            for i, goal in enumerate(goals, 1):
                f.write(f"{i}. {goal}\n")
            f.write(f"\n## Results\n\n{result}")
            
        print(f"\n📁 Complete analysis saved to: comprehensive_fab_security_analysis.md")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print("Falling back to direct tool analysis...")
        fallback_result = analyze_fab_logs_tool("")
        print("\nDirect Analysis Results:")
        print("-" * 50)
        print(fallback_result)

if __name__ == "__main__":
    main()