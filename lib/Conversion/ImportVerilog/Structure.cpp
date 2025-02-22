//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/Compilation.h"

using namespace circt;
using namespace ImportVerilog;

//===----------------------------------------------------------------------===//
// Module Member Conversion
//===----------------------------------------------------------------------===//

namespace {
struct MemberVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  MemberVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  // Skip empty members (stray semicolons).
  LogicalResult visit(const slang::ast::EmptyMemberSymbol &) {
    return success();
  }

  // Skip members that are implicitly imported from some other scope for the
  // sake of name resolution, such as enum variant names.
  LogicalResult visit(const slang::ast::TransparentMemberSymbol &) {
    return success();
  }

  // Skip typedefs.
  LogicalResult visit(const slang::ast::TypeAliasType &) { return success(); }

  // Handle instances.
  LogicalResult visit(const slang::ast::InstanceSymbol &instNode) {
    auto targetModule = context.convertModuleHeader(&instNode.body);
    if (!targetModule)
      return failure();

    builder.create<moore::InstanceOp>(
        loc, builder.getStringAttr(instNode.name),
        FlatSymbolRefAttr::get(targetModule.getSymNameAttr()));

    return success();
  }

  // Handle variables.
  LogicalResult visit(const slang::ast::VariableSymbol &varNode) {
    auto type = context.convertType(*varNode.getDeclaredType());
    if (!type)
      return failure();

    Value initial;
    if (const auto *initNode = varNode.getInitializer()) {
      return mlir::emitError(loc,
                             "variable initializer expressions not supported");
    }

    builder.create<moore::VariableOp>(
        loc, type, builder.getStringAttr(varNode.name), initial);
    return success();
  }

  /// Emit an error for all other members.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported construct: ")
        << slang::ast::toString(node.kind);
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Structure and Hierarchy Conversion
//===----------------------------------------------------------------------===//

/// Convert an entire Slang compilation to MLIR ops. This is the main entry
/// point for the conversion.
LogicalResult
Context::convertCompilation(slang::ast::Compilation &compilation) {
  const auto &root = compilation.getRoot();

  // Visit all top-level declarations in all compilation units. This does not
  // include instantiable constructs like modules, interfaces, and programs,
  // which are listed separately as top instances.
  for (auto *unit : root.compilationUnits) {
    for (const auto &member : unit->members()) {
      // Error out on all top-level declarations.
      auto loc = convertLocation(member.location);
      return mlir::emitError(loc, "unsupported construct: ")
             << slang::ast::toString(member.kind);
    }
  }

  // Prime the root definition worklist by adding all the top-level modules.
  SmallVector<const slang::ast::InstanceSymbol *> topInstances;
  for (auto *inst : root.topInstances)
    convertModuleHeader(&inst->body);

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto *module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  return success();
}

/// Convert a module and its ports to an empty module op in the IR. Also adds
/// the op to the worklist of module bodies to be lowered. This acts like a
/// module "declaration", allowing instances to already refer to a module even
/// before its body has been lowered.
moore::SVModuleOp
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  if (auto op = moduleOps.lookup(module))
    return op;
  auto loc = convertLocation(module->location);
  OpBuilder::InsertionGuard g(builder);

  // We only support modules for now. Extension to interfaces and programs
  // should be trivial though, since they are essentially the same thing with
  // only minor differences in semantics.
  if (module->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Module) {
    mlir::emitError(loc, "unsupported construct: ")
        << module->getDefinition().getKindString();
    return {};
  }

  // Handle the port list.
  for (auto *symbol : module->getPortList()) {
    auto portLoc = convertLocation(symbol->location);
    mlir::emitError(portLoc, "unsupported module port: ")
        << slang::ast::toString(symbol->kind);
    return {};
  }

  // Pick an insertion point for this module according to the source file
  // location.
  auto it = orderedRootOps.lower_bound(module->location);
  if (it == orderedRootOps.end())
    builder.setInsertionPointToEnd(intoModuleOp.getBody());
  else
    builder.setInsertionPoint(it->second);

  // Create an empty module that corresponds to this module.
  auto moduleOp = builder.create<moore::SVModuleOp>(loc, module->name);
  orderedRootOps.insert(it, {module->location, moduleOp});
  moduleOp.getBodyRegion().emplaceBlock();

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);
  moduleOps.insert({module, moduleOp});
  return moduleOp;
}

/// Convert a module's body to the corresponding IR ops. The module op must have
/// already been created earlier through a `convertModuleHeader` call.
LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  auto moduleOp = moduleOps.lookup(module);
  assert(moduleOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  for (auto &member : module->members()) {
    auto loc = convertLocation(member.location);
    if (failed(member.visit(MemberVisitor(*this, loc))))
      return failure();
  }

  return success();
}
