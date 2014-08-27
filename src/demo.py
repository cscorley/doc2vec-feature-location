from __future__ import print_function

from antlr4 import *
#from src.grammars.Java8Lexer import Java8Lexer as JavaLexer
#from src.grammars.Java8Parser import Java8Parser as JavaParser
#from src.grammars.JavaLexer import JavaLexer
#from src.grammars.JavaParser import JavaParser
import src.grammars.JavaLexer as JavaLexer
import src.grammars.JavaParser as JavaParser
import src.grammars.JavaListener as JavaListener


class MethodPrinter(JavaListener.JavaListener):
    def __init__(self):
        self.anon_count = 0
        self.anon = False

    def enterMethodDeclaration(self, ctx):
        c = ctx.parentCtx
        s = str(ctx.Identifier())
        while c:
            if hasattr(c, 'Identifier') and (
                isinstance(c, JavaParser.JavaParser.ClassDeclarationContext) or
                isinstance(c, JavaParser.JavaParser.MethodDeclarationContext)):
                if c.Identifier():
                    s = str(c.Identifier()) + '.' + s # + '{' + str(type(c)) + '}.' + s
            elif self.anon and isinstance(c, JavaParser.JavaParser.ClassBodyDeclarationContext):
                s = '$' + str(self.anon_count) + '.' + s
                self.anon = False

            c = c.parentCtx

        s += '('
        pl = ctx.formalParameters().formalParameterList()
        if pl:
            p = pl.formalParameter(0)
            i = 0
            found = False
            while p is not None:
                if found:
                    s += ','

                for j in range(p.getChildCount()):
                    c = p.getChild(j)
                    if isinstance(c, JavaParser.JavaParser.BasicTypeContext):
                        s += c.getText()

                i += 1
                p = pl.formalParameter(i)
                found = True

        s += ')'
        print(s)


    def enterClassBodyDeclaration(self, ctx):
        c = ctx.parentCtx
        self.anon = False
        while c:
            if isinstance(c, JavaParser.JavaParser.CreatorContext):
                self.anon = True

            c = c.parentCtx

        if self.anon:
            self.anon_count += 1

    # copy copy copy
    enterConstructorDeclaration = enterMethodDeclaration
    enterInterfaceMethodDeclaration = enterMethodDeclaration
    enterGenericInterfaceMethodDeclaration = enterMethodDeclaration
    enterGenericMethodDeclaration = enterMethodDeclaration


def main(argv):
    print(argv[1])
    input = FileStream(argv[1])
    lexer = JavaLexer.JavaLexer(input)
    stream = CommonTokenStream(lexer)
    parser = JavaParser.JavaParser(stream)
    tree = parser.compilationUnit()
    printer = MethodPrinter()
    walker = JavaParser.ParseTreeWalker()
    walker.walk(printer, tree)

if __name__ == '__main__':
    import sys
    main(sys.argv)
