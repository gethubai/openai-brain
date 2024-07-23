import {
  BrainPromptResponse,
  IBrainService,
  IAudioTranscriberBrainService,
  ITextBrainService,
  LocalAudioPrompt,
  TextBrainPrompt,
  IBrainPromptContext,
  BrainSettingsValidationResult,
  IImageGenerationBrainService,
  ImageGenerationBrainPrompt,
  ResponseFile,
  FileAttachment,
} from '@hubai/brain-sdk';
import OpenAI from 'openai';
import fs from 'fs';
import fetch from 'node-fetch';
import { TranscriptionCreateParams } from 'openai/resources/audio/transcriptions.mjs';

/* Example of setting required by this brain */
export interface ISettings {
  apiKey: string;
  textModel?: string;
  audioTranscriberModel?: string;
  audioTranscriberDefaultLanguage?: string;
  imageGenerationSize: '256x256' | '512x512' | '1024x1024';
  imageGenerationCount: string;
  maxCharactersHistorySize: number;
  imageDetail: 'low' | 'high';
  imageGenerationModel: string;
}

/* If your brain does not support AudioTranscription just remove the interface implementation */
export default class MyBrainService
  implements
    IBrainService,
    ITextBrainService<ISettings>,
    IAudioTranscriberBrainService<ISettings>,
    IImageGenerationBrainService<ISettings>
{
  openAI?: OpenAI;
  currentKey?: string;

  async transcribeAudio(
    prompt: LocalAudioPrompt,
    context: IBrainPromptContext<ISettings>,
  ): Promise<BrainPromptResponse> {
    // First we validate the settings
    const validationResult = this.validateSettings(context.settings);

    // If the settings are not valid we return the validation result
    if (!validationResult.success) {
      return Promise.resolve({
        result: validationResult.getMessage(),
        validationResult,
      });
    }

    const fileStream = fs.createReadStream(prompt.audioFilePath);

    // We create the params to send to the API
    const params: TranscriptionCreateParams = {
      file: fileStream, // The path to the audio file
      language:
        prompt.language ||
        context.settings.audioTranscriberDefaultLanguage ||
        'en', // The language of the audio file
      model: context.settings.audioTranscriberModel, // The model to use for the transcription
    };

    // We send the request to the API
    const result = await this.getClient(
      context.settings,
    ).audio.transcriptions.create(params);

    // We return the result
    return { result: result.text, validationResult };
  }

  async generateImage(
    prompts: ImageGenerationBrainPrompt[],
    context: IBrainPromptContext<ISettings>,
  ): Promise<BrainPromptResponse> {
    const prompt = prompts[prompts.length - 1]; // OpenAI API does not support multiple prompts for DALL-E image generation, so we just get the latest one
    // First we validate the settings
    const validationResult = this.validateSettings(context.settings);

    // If the settings are not valid we return the validation result
    if (!validationResult.success) {
      return Promise.resolve({
        result: validationResult.getMessage(),
        validationResult,
      });
    }

    let responseFormat = 'url';
    // Check if the prompt expects a base64 encoded image as response
    if (prompt.expectedResponseType === 'base64') {
      responseFormat = 'b64_json';
    }

    const textModels = {
      'Dall-E 2': 'dall-e-2',
      'Dall-E 3': 'dall-e-3',
    };

    const model = textModels[context.settings.imageGenerationModel];

    const params: OpenAI.Images.ImageGenerateParams = {
      model,
      prompt: prompt.message.trim(),
      n:
        model === 'dall-e-3'
          ? 1
          : Number.parseInt(context.settings.imageGenerationCount), // Dall-e 3 only supports 1 image
      size:
        model === 'dall-e-3'
          ? '1024x1024'
          : context.settings.imageGenerationSize,
      user: context.senderId,
      response_format: responseFormat as any,
    };

    // Call the API
    const result = await this.getClient(context.settings).images.generate(
      params,
    );

    // Get the urls from the response
    const urls = result.data.map((d) =>
      responseFormat === 'url' ? d.url : d.b64_json,
    );

    const attachments: ResponseFile[] = [];

    // Parse the attachments
    for (const url of urls) {
      let data: Buffer | string = url;
      // If the prompt expects a binary response we fetch the image and return it as a buffer
      if (prompt.expectedResponseType === 'binary') {
        data = await (await fetch(url)).buffer();
      }

      attachments.push({
        data,
        fileType: 'image',
        mimeType: 'image/png', // mimeType is always png for DALL-E
      });
    }

    return {
      result: '', // If you want to return any additional text you can do it here
      attachments, // Attachments are returned as a list of files
      validationResult,
    };
  }

  sendTextPrompt(
    prompts: TextBrainPrompt[],
    context: IBrainPromptContext<ISettings>,
  ): Promise<BrainPromptResponse> {
    // First we validate the settings
    const validationResult = this.validateSettings(context.settings);

    // If the settings are not valid we return the validation result
    if (!validationResult.success) {
      return Promise.resolve({
        result: validationResult.getMessage(),
        validationResult,
      });
    }
    const maxCharactersHistorySize =
      context.settings?.maxCharactersHistorySize ?? 3000;

    const messagesTruncated = this.truncateStringList(
      prompts,
      maxCharactersHistorySize,
    );

    const lastPrompt = messagesTruncated[messagesTruncated.length - 1];
    const hasImageAttached = messagesTruncated.filter(
      (m) =>
        m.attachments?.filter((a) => a.mimeType.startsWith('image/')).length,
    ).length;

    const getMessageContent = (
      message: TextBrainPrompt,
    ): string | OpenAI.Chat.ChatCompletionContentPart[] => {
      if (
        !message.attachments?.filter((a) => a.mimeType.startsWith('image/'))
          .length || // Check if has any image attachment
        message.role === 'brain' // Check if the message is from the brain
      ) {
        return message.message;
      }

      return [
        {
          type: 'text',
          text: message.message,
        },
      ].concat(
        message.attachments.map((m) => ({
          type: 'image_url',
          image_url: {
            url: this.convertImageAttachmentToBase64(m),
            detail:
              message.message === lastPrompt.message
                ? context.settings.imageDetail
                : 'low',
          },
        })) as any,
      ) as OpenAI.Chat.ChatCompletionContentPart[];
    };

    const textModels = {
      'GPT 4o': 'gpt-4o',
      'GPT 4o-mini': 'gpt-4o-mini',
      vision: 'gpt-4o-mini',
    };

    // We create the params to send to the API
    const params: OpenAI.Chat.ChatCompletionCreateParams = {
      model: hasImageAttached
        ? textModels['vision']
        : textModels[context.settings.textModel], //context.settings.textModel,
      messages: messagesTruncated.map((m) => ({
        role: m.role === 'brain' ? 'assistant' : m.role,
        content: getMessageContent(m) as any,
      })),
      max_tokens: maxCharactersHistorySize,
    };

    return this.getClient(context.settings)
      .chat.completions.create(params)
      .then((response: OpenAI.Chat.ChatCompletion) => ({
        // Return the OpenAI text response in the result field
        result: response.choices[0].message.content.trim(),
        validationResult,
      }));
  }

  convertImageAttachmentToBase64(attachment: FileAttachment): string {
    const imageBase64 = fs.readFileSync(attachment.path).toString('base64');
    const imageUrl = `data:${attachment.mimeType};base64,${imageBase64}`;

    return imageUrl;
  }

  truncateStringList(
    list: TextBrainPrompt[],
    maxCharacters: number,
  ): TextBrainPrompt[] {
    if (list.length <= 2) {
      return list;
    }

    let totalCharacters = list.map((m) => m.message).join('').length;

    while (totalCharacters > maxCharacters && list.length > 1) {
      const firstItem = list.shift();
      if (firstItem) {
        totalCharacters -= firstItem.message.length;
      } else {
        break;
      }
    }

    return list;
  }

  validateSettings(settings: ISettings): BrainSettingsValidationResult {
    const validation = new BrainSettingsValidationResult();

    const content = `# OpenAI API Key is Missing
    Ops! Looks like you didn't configure your OpenAI API Key yet. This is required to use this brain.

    ## How to get an OpenAI API Key

    [Log in into your Open AI account](https://platform.openai.com/login?launch) or [create one](https://chat.openai.com/auth/login) if you don't have it.
    After that, go to the [API Key section of Open AI dashboard](https://platform.openai.com/api-keys) and copy your API key (or generate one if you still don't have).

    ![Open AI Dashboard](https://www.hubai.app/assets/images/openaidashboard-a5eec4bd5881c386268c5f7e8a02cdbb.png)

    ## How to configure your OpenAI API Key at HubAI

    Go to the Brains page, select this brain and set the OpenAI API Key. After that just click on the **Save Settings** button and you're ready to go!
    ![brain settings](https://www.hubai.app/assets/images/brain-settings-40bc90772d6b2ab68a8ab4a255cb4987.png)

    ## How to use gpt-4 and Dall-e-3 premium models
In order to use all the features that Chat GPT brain has to offer (like gpt-4), we recommend you to buy at least $0.5 cents of credits on Open AI. You can do that [by clicking here](https://platform.openai.com/account/billing/overview). You only have to do this once.
    `;

    if (!settings?.apiKey || settings.apiKey.length < 10) {
      validation.addError(content);
    }

    return validation;
  }

  getClient(settings: ISettings): OpenAI {
    if (!this.openAI || this.currentKey !== settings.apiKey) {
      this.openAI = new OpenAI({ apiKey: settings.apiKey });
      this.currentKey = settings.apiKey;
    }
    return this.openAI;
  }
}
