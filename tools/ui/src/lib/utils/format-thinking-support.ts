import type { ApiProps } from '$lib/types/api';

/**
 * Whether the served chat format has a thinking channel.
 *
 * Replaces `chat-template-thinking-detector`, which answered this by
 * string-searching the chat template for thinking markers. That module was
 * deleted along with the Jinja engine, but its import was left behind in
 * `models.svelte.ts`, so `npm run build` failed on a missing file -- invisible
 * until the package build ran npm the way upstream Arch does.
 *
 * Reinstating the search was never an option: this build has no chat template to
 * search, and inferring a format's abilities from template text is exactly what
 * the server side spent this fork removing. The server states the fact instead,
 * in `/props` -> `chat_template_caps.supports_thinking`, and the UI reads it.
 *
 * Undeclared means "assume yes": an older server that predates the capability
 * would otherwise have its thinking controls hidden, which is a worse failure
 * than showing a control that turns out to do nothing.
 */
export function detectThinkingSupport(props: ApiProps | undefined | null): boolean {
	return props?.chat_template_caps?.supports_thinking ?? true;
}

/**
 * The same answer with a reason string, for the settings UI and for debugging.
 */
export function detectThinkingSupportWithReason(props: ApiProps | undefined | null): {
	supported: boolean;
	reason: string;
} {
	if (!props) {
		return { supported: false, reason: 'Server properties not loaded yet' };
	}

	const declared = props.chat_template_caps?.supports_thinking;

	if (declared === undefined) {
		return {
			supported: true,
			reason: 'Server does not declare supports_thinking; assuming supported'
		};
	}

	return declared
		? { supported: true, reason: 'Chat format declares a thinking channel' }
		: { supported: false, reason: 'Chat format declares no thinking channel' };
}
