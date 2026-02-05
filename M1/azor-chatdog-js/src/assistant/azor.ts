/**
 * Azor assistant factory (backward compatibility)
 */

import type { Assistant } from './assistant.js';
import { getDefaultAssistant } from './registry.js';

/**
 * Create the Azor assistant with predefined personality
 */
export function createAzorAssistant(): Assistant {
  return getDefaultAssistant();
}
