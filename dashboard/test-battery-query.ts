// Test script to verify Kalman filter is used in battery performance queries
import { askExecutive } from './lib/agent';

async function testBatteryQuery() {
  console.log('=== Testing Battery Performance Query with Kalman Filter ===\n');
  
  const testQueries = [
    "Show me IoT battery performance analysis",
    "Show me battery performance",
    "What is the battery status?",
    "Show me IoT battery metrics"
  ];
  
  for (const query of testQueries) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Testing query: "${query}"`);
    console.log('='.repeat(60));
    
    try {
      const result = await askExecutive(query, { userRole: 'executive' });
      
      console.log('\n✅ Query executed successfully');
      console.log('Summary:', result.summary);
      
      if (result.data && result.data.length > 0) {
        console.log(`Data items: ${result.data.length}`);
      }
      
      // Check if battery data is in the response
      const summaryText = result.summary.toLowerCase();
      if (summaryText.includes('battery') || summaryText.includes('voltage') || summaryText.includes('capacity')) {
        console.log('✅ Battery data detected in response');
      }
      
    } catch (error) {
      console.error('❌ Error:', error);
    }
  }
  
  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
  console.log('\nNote: Check the logs above for:');
  console.log('  [Agent] Fetching battery performance with Kalman filter enabled');
  console.log('  [Kalman Filter] Starting filtering process');
  console.log('  [Battery Analytics] Processing device: ...');
}

// Run the test
testBatteryQuery().catch(console.error);
