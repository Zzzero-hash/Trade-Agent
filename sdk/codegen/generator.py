"""
Code generation tools for multiple programming languages
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from jinja2 import Environment, FileSystemLoader, Template
import yaml


class Language(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: str
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.responses is None:
            self.responses = {}
        if self.tags is None:
            self.tags = []


@dataclass
class DataModel:
    """Data model definition"""
    name: str
    description: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str] = None
    
    def __post_init__(self):
        if self.required is None:
            self.required = []


@dataclass
class APISpec:
    """Complete API specification"""
    title: str
    version: str
    description: str
    base_url: str
    endpoints: List[APIEndpoint]
    models: List[DataModel]
    auth_methods: List[str] = None
    
    def __post_init__(self):
        if self.auth_methods is None:
            self.auth_methods = ["bearer"]


class CodeGenerator:
    """Generates SDK code for multiple programming languages"""
    
    def __init__(self, templates_dir: str = None):
        if templates_dir is None:
            templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        
        self.templates_dir = templates_dir
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['camel_case'] = self._camel_case
        self.env.filters['pascal_case'] = self._pascal_case
        self.env.filters['snake_case'] = self._snake_case
        self.env.filters['kebab_case'] = self._kebab_case
        self.env.filters['type_mapping'] = self._type_mapping
    
    def generate_sdk(self, api_spec: APISpec, language: Language, 
                    output_dir: str, config: Dict[str, Any] = None) -> None:
        """Generate SDK for specified language"""
        if config is None:
            config = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate based on language
        if language == Language.PYTHON:
            self._generate_python_sdk(api_spec, output_dir, config)
        elif language == Language.TYPESCRIPT:
            self._generate_typescript_sdk(api_spec, output_dir, config)
        elif language == Language.JAVA:
            self._generate_java_sdk(api_spec, output_dir, config)
        elif language == Language.CSHARP:
            self._generate_csharp_sdk(api_spec, output_dir, config)
        elif language == Language.GO:
            self._generate_go_sdk(api_spec, output_dir, config)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _generate_python_sdk(self, api_spec: APISpec, output_dir: str, config: Dict[str, Any]):
        """Generate Python SDK"""
        package_name = config.get("package_name", "ai_trading_platform_client")
        
        # Generate package structure
        package_dir = os.path.join(output_dir, package_name)
        os.makedirs(package_dir, exist_ok=True)
        
        # Generate __init__.py
        init_template = self.env.get_template("python/__init__.py.j2")
        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write(init_template.render(
                api_spec=api_spec,
                package_name=package_name,
                config=config
            ))
        
        # Generate models
        models_template = self.env.get_template("python/models.py.j2")
        with open(os.path.join(package_dir, "models.py"), "w") as f:
            f.write(models_template.render(
                api_spec=api_spec,
                models=api_spec.models
            ))
        
        # Generate client
        client_template = self.env.get_template("python/client.py.j2")
        with open(os.path.join(package_dir, "client.py"), "w") as f:
            f.write(client_template.render(
                api_spec=api_spec,
                endpoints=api_spec.endpoints
            ))
        
        # Generate exceptions
        exceptions_template = self.env.get_template("python/exceptions.py.j2")
        with open(os.path.join(package_dir, "exceptions.py"), "w") as f:
            f.write(exceptions_template.render(api_spec=api_spec))
        
        # Generate setup.py
        setup_template = self.env.get_template("python/setup.py.j2")
        with open(os.path.join(output_dir, "setup.py"), "w") as f:
            f.write(setup_template.render(
                api_spec=api_spec,
                package_name=package_name,
                config=config
            ))
        
        # Generate requirements.txt
        requirements_template = self.env.get_template("python/requirements.txt.j2")
        with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
            f.write(requirements_template.render(config=config))
    
    def _generate_typescript_sdk(self, api_spec: APISpec, output_dir: str, config: Dict[str, Any]):
        """Generate TypeScript SDK"""
        package_name = config.get("package_name", "ai-trading-platform-client")
        
        # Generate package.json
        package_template = self.env.get_template("typescript/package.json.j2")
        with open(os.path.join(output_dir, "package.json"), "w") as f:
            f.write(package_template.render(
                api_spec=api_spec,
                package_name=package_name,
                config=config
            ))
        
        # Generate tsconfig.json
        tsconfig_template = self.env.get_template("typescript/tsconfig.json.j2")
        with open(os.path.join(output_dir, "tsconfig.json"), "w") as f:
            f.write(tsconfig_template.render(config=config))
        
        # Create src directory
        src_dir = os.path.join(output_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        
        # Generate types
        types_template = self.env.get_template("typescript/types.ts.j2")
        with open(os.path.join(src_dir, "types.ts"), "w") as f:
            f.write(types_template.render(
                api_spec=api_spec,
                models=api_spec.models
            ))
        
        # Generate client
        client_template = self.env.get_template("typescript/client.ts.j2")
        with open(os.path.join(src_dir, "client.ts"), "w") as f:
            f.write(client_template.render(
                api_spec=api_spec,
                endpoints=api_spec.endpoints
            ))
        
        # Generate index
        index_template = self.env.get_template("typescript/index.ts.j2")
        with open(os.path.join(src_dir, "index.ts"), "w") as f:
            f.write(index_template.render(api_spec=api_spec))
    
    def _generate_java_sdk(self, api_spec: APISpec, output_dir: str, config: Dict[str, Any]):
        """Generate Java SDK"""
        package_name = config.get("package_name", "com.aitradingplatform.client")
        package_path = package_name.replace(".", "/")
        
        # Create package directory
        src_dir = os.path.join(output_dir, "src", "main", "java", package_path)
        os.makedirs(src_dir, exist_ok=True)
        
        # Generate models
        for model in api_spec.models:
            model_template = self.env.get_template("java/Model.java.j2")
            with open(os.path.join(src_dir, f"{model.name}.java"), "w") as f:
                f.write(model_template.render(
                    model=model,
                    package_name=package_name
                ))
        
        # Generate client
        client_template = self.env.get_template("java/Client.java.j2")
        with open(os.path.join(src_dir, "TradingPlatformClient.java"), "w") as f:
            f.write(client_template.render(
                api_spec=api_spec,
                package_name=package_name,
                endpoints=api_spec.endpoints
            ))
        
        # Generate pom.xml
        pom_template = self.env.get_template("java/pom.xml.j2")
        with open(os.path.join(output_dir, "pom.xml"), "w") as f:
            f.write(pom_template.render(
                api_spec=api_spec,
                package_name=package_name,
                config=config
            ))
    
    def _generate_csharp_sdk(self, api_spec: APISpec, output_dir: str, config: Dict[str, Any]):
        """Generate C# SDK"""
        namespace = config.get("namespace", "AiTradingPlatform.Client")
        
        # Generate models
        for model in api_spec.models:
            model_template = self.env.get_template("csharp/Model.cs.j2")
            with open(os.path.join(output_dir, f"{model.name}.cs"), "w") as f:
                f.write(model_template.render(
                    model=model,
                    namespace=namespace
                ))
        
        # Generate client
        client_template = self.env.get_template("csharp/Client.cs.j2")
        with open(os.path.join(output_dir, "TradingPlatformClient.cs"), "w") as f:
            f.write(client_template.render(
                api_spec=api_spec,
                namespace=namespace,
                endpoints=api_spec.endpoints
            ))
        
        # Generate project file
        csproj_template = self.env.get_template("csharp/Client.csproj.j2")
        with open(os.path.join(output_dir, "AiTradingPlatform.Client.csproj"), "w") as f:
            f.write(csproj_template.render(
                api_spec=api_spec,
                config=config
            ))
    
    def _generate_go_sdk(self, api_spec: APISpec, output_dir: str, config: Dict[str, Any]):
        """Generate Go SDK"""
        module_name = config.get("module_name", "github.com/ai-trading-platform/go-client")
        
        # Generate go.mod
        gomod_template = self.env.get_template("go/go.mod.j2")
        with open(os.path.join(output_dir, "go.mod"), "w") as f:
            f.write(gomod_template.render(
                module_name=module_name,
                config=config
            ))
        
        # Generate types
        types_template = self.env.get_template("go/types.go.j2")
        with open(os.path.join(output_dir, "types.go"), "w") as f:
            f.write(types_template.render(
                api_spec=api_spec,
                models=api_spec.models
            ))
        
        # Generate client
        client_template = self.env.get_template("go/client.go.j2")
        with open(os.path.join(output_dir, "client.go"), "w") as f:
            f.write(client_template.render(
                api_spec=api_spec,
                endpoints=api_spec.endpoints
            ))
    
    def load_openapi_spec(self, spec_file: str) -> APISpec:
        """Load API specification from OpenAPI file"""
        with open(spec_file, 'r') as f:
            if spec_file.endswith('.yaml') or spec_file.endswith('.yml'):
                spec_data = yaml.safe_load(f)
            else:
                spec_data = json.load(f)
        
        # Parse OpenAPI spec
        title = spec_data.get('info', {}).get('title', 'API Client')
        version = spec_data.get('info', {}).get('version', '1.0.0')
        description = spec_data.get('info', {}).get('description', '')
        
        # Extract base URL
        servers = spec_data.get('servers', [])
        base_url = servers[0]['url'] if servers else 'https://api.example.com'
        
        # Parse endpoints
        endpoints = []
        paths = spec_data.get('paths', {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    endpoint = APIEndpoint(
                        path=path,
                        method=method.upper(),
                        name=details.get('operationId', f"{method}_{path.replace('/', '_')}"),
                        description=details.get('summary', ''),
                        parameters=details.get('parameters', []),
                        request_body=details.get('requestBody'),
                        responses=details.get('responses', {}),
                        tags=details.get('tags', [])
                    )
                    endpoints.append(endpoint)
        
        # Parse models
        models = []
        components = spec_data.get('components', {})
        schemas = components.get('schemas', {})
        for name, schema in schemas.items():
            model = DataModel(
                name=name,
                description=schema.get('description', ''),
                properties=schema.get('properties', {}),
                required=schema.get('required', [])
            )
            models.append(model)
        
        return APISpec(
            title=title,
            version=version,
            description=description,
            base_url=base_url,
            endpoints=endpoints,
            models=models
        )
    
    # Utility methods for template filters
    def _camel_case(self, text: str) -> str:
        """Convert to camelCase"""
        components = text.replace('-', '_').split('_')
        return components[0] + ''.join(word.capitalize() for word in components[1:])
    
    def _pascal_case(self, text: str) -> str:
        """Convert to PascalCase"""
        components = text.replace('-', '_').split('_')
        return ''.join(word.capitalize() for word in components)
    
    def _snake_case(self, text: str) -> str:
        """Convert to snake_case"""
        return text.replace('-', '_').lower()
    
    def _kebab_case(self, text: str) -> str:
        """Convert to kebab-case"""
        return text.replace('_', '-').lower()
    
    def _type_mapping(self, openapi_type: str, language: Language) -> str:
        """Map OpenAPI types to language-specific types"""
        mappings = {
            Language.PYTHON: {
                'string': 'str',
                'integer': 'int',
                'number': 'float',
                'boolean': 'bool',
                'array': 'List',
                'object': 'Dict'
            },
            Language.TYPESCRIPT: {
                'string': 'string',
                'integer': 'number',
                'number': 'number',
                'boolean': 'boolean',
                'array': 'Array',
                'object': 'object'
            },
            Language.JAVA: {
                'string': 'String',
                'integer': 'Integer',
                'number': 'Double',
                'boolean': 'Boolean',
                'array': 'List',
                'object': 'Object'
            },
            Language.CSHARP: {
                'string': 'string',
                'integer': 'int',
                'number': 'double',
                'boolean': 'bool',
                'array': 'List',
                'object': 'object'
            },
            Language.GO: {
                'string': 'string',
                'integer': 'int',
                'number': 'float64',
                'boolean': 'bool',
                'array': '[]',
                'object': 'map[string]interface{}'
            }
        }
        
        return mappings.get(language, {}).get(openapi_type, openapi_type)


# CLI interface
def main():
    """Command-line interface for code generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SDK code from OpenAPI specification")
    parser.add_argument("spec_file", help="OpenAPI specification file (JSON or YAML)")
    parser.add_argument("language", choices=[lang.value for lang in Language], help="Target language")
    parser.add_argument("output_dir", help="Output directory for generated code")
    parser.add_argument("--config", help="Configuration file (JSON or YAML)")
    parser.add_argument("--package-name", help="Package/module name")
    parser.add_argument("--namespace", help="Namespace (for C#)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Override with command-line arguments
    if args.package_name:
        config['package_name'] = args.package_name
    if args.namespace:
        config['namespace'] = args.namespace
    
    # Generate SDK
    generator = CodeGenerator()
    api_spec = generator.load_openapi_spec(args.spec_file)
    generator.generate_sdk(api_spec, Language(args.language), args.output_dir, config)
    
    print(f"SDK generated successfully in {args.output_dir}")


if __name__ == "__main__":
    main()