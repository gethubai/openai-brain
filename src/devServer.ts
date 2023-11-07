/* DO NOT DELETE OR CHANGE THIS FILE, ITS FOR DEVELOPMENT PURPOSES ONLY */

import express from 'express';
import bodyParser from 'body-parser';
import brain from './main';

const host = '127.0.0.1';
const port = 1367;
const app = express();
app.use(bodyParser.json())

app.get('/', function (_, response) {
  return response.send('Brain dev server is running');
});

app.post('/api/transcribeAudio', async function (request, response) {
  const {audioPath, context} = request.body;

  const result = await brain.transcribeAudio(audioPath, context);
  return response.json(result);
});

app.post('/api/textPrompt', async function (request, response) {
  const {prompts, context} = request.body;

  const result = await brain.sendTextPrompt(prompts, context);
  return response.json(result);
});

app.post('/api/imageGeneration', async function (request, response) {
  const {prompts, context} = request.body;

  const result = await brain.generateImage(prompts, context);
  return response.json(result);
});

app.get('/stream', function (_, res, __) {
  //when using text/plain it did not stream
  //without charset=utf-8, it only worked in Chrome, not Firefox
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.setHeader('Transfer-Encoding', 'chunked');

  res.write("Thinking...");
  sendAndSleep(res, 1);
});


const sendAndSleep = function (response, counter) {
  if (counter > 10) {
    response.end();
  } else {
    response.write(" ;i=" + counter);
    counter++;
    setTimeout(function () {
      sendAndSleep(response, counter);
    }, 1000)
  };
};

app.listen(port, host);
