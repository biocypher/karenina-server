# MCP Server Presets

This directory contains MCP (Model Context Protocol) server preset configurations. Each preset is a JSON file that defines an MCP server connection.

## Overview

MCP presets allow you to configure external MCP servers that provide additional tools and context for LLM-based operations. Presets are stored as individual JSON files in this directory and can be managed through the API or by editing the files directly.

## Directory Location

By default, this directory is located at `mcp_presets/` in the current working directory. You can change this location by setting the `MCP_PRESETS_DIR` environment variable to point to a parent directory (the `mcp_presets/` subdirectory will be created inside).

Example:
```bash
export MCP_PRESETS_DIR=/path/to/your/config
# This will use /path/to/your/config/mcp_presets/
```

## Preset File Format

Each preset is a JSON file with the following structure:

```json
{
  "name": "server-name",
  "url": "http://localhost:8000/mcp",
  "tools": ["tool1", "tool2"]
}
```

### Fields

- **name** (required): A unique identifier for this MCP server
- **url** (required): The URL endpoint for the MCP server
- **tools** (optional): An array of tool names to filter from this server. If omitted, all tools from the server will be available.

### File Naming

Preset files should have a `.json` extension. The filename is typically derived from the preset name (lowercase, spaces replaced with underscores), but any valid filename works as long as the JSON structure is correct.

Examples:
- `biocontext.json`
- `my_custom_server.json`
- `research_assistant.json`

## Usage

### Through the API

**List all presets:**
```bash
curl http://localhost:8000/api/get-mcp-preset-configs
```

**Save a new preset:**
```bash
curl -X POST http://localhost:8000/api/save-mcp-preset \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-server",
    "url": "http://localhost:9000/mcp",
    "tools": ["search", "summarize"]
  }'
```

**Delete a preset:**
```bash
curl -X POST http://localhost:8000/api/delete-mcp-preset \
  -H "Content-Type: application/json" \
  -d '{"name": "my-server"}'
```

### Manual File Management

You can also create, edit, or delete preset files directly in this directory. The server will automatically discover and load all `.json` files on startup or when the API is called.

**Create a new preset:**
1. Create a new `.json` file in this directory
2. Add the required `name` and `url` fields
3. Optionally add a `tools` array to filter specific tools
4. Save the file

**Edit an existing preset:**
1. Open the preset file
2. Modify the fields as needed
3. Save the file

**Delete a preset:**
1. Remove the corresponding `.json` file from this directory

## Security Notes

### Path Validation

The MCP preset service validates all directory paths to prevent directory traversal attacks. Allowed paths are:
- Within the project directory
- Within the user's home directory
- Within the system temp directory (for tests)

Attempts to access forbidden directories (e.g., `/etc`, `/usr`, `/bin`) will be rejected.

### File Permissions

When creating preset files through the API, they are created with permissions `0o644` (readable by everyone, writable only by owner).

### Input Sanitization

- Preset names are sanitized to prevent filename injection attacks
- Only alphanumeric characters, spaces, hyphens, and underscores are allowed in names
- Names are limited to 100 characters

## Examples

See `example_biocontext.json` in this directory for a sample preset configuration.

## Troubleshooting

### Preset not appearing in API response

- Check that the file has a `.json` extension
- Verify the JSON is valid (use `jq` or a JSON validator)
- Ensure the file contains both `name` and `url` fields
- Check server logs for parsing errors

### Cannot save preset through API

- Verify the `name` and `url` fields are non-empty strings
- If providing `tools`, ensure it's an array of strings
- Check that you have write permissions to the `mcp_presets/` directory

### Directory not found errors

- Ensure the `mcp_presets/` directory exists
- If using `MCP_PRESETS_DIR`, verify the path is correct and accessible
- Check that the path is within allowed directories (not in forbidden system directories)

## Migration from Environment Variable

**Previous behavior:** MCP server configurations were provided via the `MCP_CONFIG` environment variable.

**Current behavior:** All MCP server configurations are now file-based presets in this directory.

To migrate from environment variable configuration:
1. Parse your old `MCP_CONFIG` JSON
2. Create individual preset files for each server
3. Remove the `MCP_CONFIG` environment variable
4. Restart the server

## Development

When developing or testing, you can use a custom presets directory:

```bash
# Use a test directory
export MCP_PRESETS_DIR=/tmp/test-mcp-presets

# Start the server
python -m karenina_server
```

This allows you to test preset configurations without affecting your production setup.
