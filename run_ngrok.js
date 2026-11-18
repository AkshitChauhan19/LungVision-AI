const ngrok = require('@ngrok/ngrok');
const fs = require('fs');

(async () => {
  const authtoken = process.env.NGROK_AUTHTOKEN;
  if (!authtoken) {
    console.error('[!] Missing NGROK_AUTHTOKEN environment variable.');
    console.error('    Set it before running: export NGROK_AUTHTOKEN="your_token_here"');
    process.exit(1);
  }

  try {
    console.log('[*] Connecting to ngrok...');
    const listener = await ngrok.forward({
      addr: process.env.PORT || 8000,
      authtoken: authtoken
    });

    const url = listener.url();
    console.log('\n' + '='.repeat(60));
    console.log('  LUNGVISION AI IS LIVE ON NGROK:');
    console.log('  >>> ' + url + ' <<<');
    console.log('='.repeat(60) + '\n');

    fs.writeFileSync('ngrok_url.txt', url);

    // Keep running indefinitely
    process.stdin.resume();
  } catch (err) {
    console.error('[!] Ngrok error:', err);
    process.exit(1);
  }
})();
