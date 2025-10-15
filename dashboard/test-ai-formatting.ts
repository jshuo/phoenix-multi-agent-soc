#!/usr/bin/env tsx
/**
 * Test AI Formatting Improvements
 * Tests the enhanced summary formatting for battery analytics
 */

import 'dotenv/config';
import { askExecutive } from './lib/agent';

async function testFormatting() {
  console.log('🧪 Testing AI Formatting Improvements\n');
  console.log('Query: "Show me IoT battery performance analysis"\n');
  console.log('=' .repeat(80));
  
  try {
    const result = await askExecutive('Show me IoT battery performance analysis', {});
    
    console.log('\n📊 AI ANALYSIS SUMMARY:\n');
    console.log(result.summary);
    console.log('\n' + '='.repeat(80));
    
    console.log('\n📈 Key Metrics:');
    console.log(`  - Devices analyzed: ${result.data?.length || 0}`);
    console.log(`  - Sources: ${result.sources?.join(', ')}`);
    console.log(`  - Recommendations: ${result.recommendations?.length || 0}`);
    
    if (result.recommendations) {
      console.log('\n💡 Recommendations:');
      result.recommendations.forEach((rec, i) => {
        console.log(`  ${i + 1}. ${rec}`);
      });
    }
    
    console.log('\n✅ Test completed successfully!');
  } catch (error) {
    console.error('\n❌ Test failed:', error);
    process.exit(1);
  }
}

testFormatting();
