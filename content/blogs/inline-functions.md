---
title: "Understanding Programs at the Lowest Level"
date: 2026-01-04T20:17:00+09:00
slug: InlineFunction
category: InlineFunction
tags:
  - AI
  - Machine-Learning
  - Graph-Neural-Networks
summary:
description:
cover:
  image: "covers/inline_function_2.png"
  alt:
  caption:
  relative: true
showtoc: true
draft: false
---

## Why Programs Felt Like Magic to Me

---

<!-- When I was younger, programs felt less like things humans built and more like things that simply existed. You wrote some words into a file, pressed a button, and suddenly the computer did something. A game ran. A window opened. A result appeared. Somewhere in between, something invisible happened - and that “something” was never really explained.

I knew, in theory, that a program was “code” and that the computer was “executing” it. But those words didn’t form a mental picture. What does executing mean? Who is doing the executing? How does a machine that only understands electricity and circuits follow instructions that look like English? And perhaps the strangest question of all: where does the program go when it runs?

At a surface level, it’s easy to accept hand-wavy answers. The computer “runs the program.” The CPU “processes instructions.” Memory “stores data.” But for a long time, those explanations felt like placeholders rather than answers. They named the parts without showing how they fit together.

The confusion became even stronger once functions entered the picture. Functions are described as neat, self-contained blocks of logic. You call one, it does its thing, and then execution continues as if nothing special happened. But that raises uncomfortable questions. How does the computer know where to come back to? What happens if one function calls another? Or if a function calls itself?

Somewhere deep down, it felt like the machine must be keeping track of all this context in a way that bordered on intelligence. Like it had a memory of intent: “I’ll go over there for a moment, then return here.” That kind of behavior feels natural to humans - but for a machine that supposedly just executes one instruction after another, it’s oddly sophisticated.

That gap - between the simple story we’re told and the complex behavior we observe - is where the feeling of magic comes from. Not the good kind of magic, but the kind that makes systems feel opaque and unapproachable. You can use them, but you don’t really understand them.

What eventually dissolved that sense of magic wasn’t learning more syntax or higher-level abstractions. It was learning how little the CPU actually does. No planning. No remembering intent. Just fetching an instruction, executing it, and moving on. Everything else - functions, calls, data structures, even “returning” from somewhere - is built on top of that simplicity.

Understanding how something as mundane as jumping to another address and storing a number on a stack can explain all of this was the moment programs stopped feeling like magic - and started feeling like carefully engineered illusions. -->

When I was younger, programs felt less like things humans built and more like things that simply existed. You wrote some words into a file, pressed a button, and suddenly the computer did something. A game ran. A window opened. A result appeared. Somewhere in between, something invisible happened, and that “something” was never really explained. I knew, in theory, that a program was “code” and that the computer was “executing” it, but those words never formed a real mental picture. What does executing actually mean? Who is doing the executing? How does a machine made of electricity and circuits follow instructions that resemble English? And perhaps the strangest question of all: where does the program go when it runs? At a surface level, it’s easy to accept vague explanations. The computer “runs the program.” The CPU “processes instructions.” Memory “stores data.” But for a long time, those answers felt more like labels than explanations. They named the parts without showing how they fit together.

The confusion deepened once functions entered the picture. Functions are described as neat, self-contained blocks of logic: you call one, it does its work, and execution continues as if nothing unusual happened. But that raises uncomfortable questions. How does the computer know where to return afterward? What happens if one function calls another, or if a function calls itself? Somewhere deep down, it felt like the machine must be keeping track of context in a way that bordered on intelligence, almost as if it remembered intent: “I’ll go over there for a moment, then come back here.” That kind of behavior feels natural to humans, but for a machine that supposedly just executes one instruction after another, it seems oddly sophisticated. That gap, between the simple story we’re told and the complex behavior we observe, is where the feeling of magic comes from. Not the inspiring kind of magic, but the kind that makes systems feel opaque and unapproachable. You can use them, but you don’t truly understand them.

What eventually dissolved that sense of magic wasn’t learning more syntax or climbing further up the abstraction ladder. It was learning how little the CPU actually does. No planning. No awareness. No memory of intent. Just fetching an instruction, executing it, and moving on. Everything else - functions, calls, data structures, even the idea of “returning” from somewhere - emerges from that simplicity. Understanding how something as mundane as jumping to another address and storing a number on a stack can explain all of this was the moment programs stopped feeling supernatural. They started feeling like carefully engineered illusions: immensely powerful, sometimes beautiful, but built entirely from simple mechanical rules repeated billions of times per second.

## Program at the Lowest Level

---

At its core, a program is not an idea, a workflow, or a set of intentions. It’s not even _logic_ in the way we usually think about it. **{{< mathjax/inline>}}<span style="color: #0084a5;">At the lowest level, a program is just data - a long sequence of numbers stored in memory.</span>{{< /mathjax/inline>}}** Those numbers are interpreted as **instructions**.

A CPU doesn’t see variables, functions, loops, or conditionals. It sees only a stream of encoded commands like _“move this value,” “add these numbers,”_ or _“jump to that address.”_ Each instruction is small, mechanical, and boring on its own. There is no understanding of _why_ the instruction exists or what it contributes to the larger program. The CPU simply executes it because that’s what it’s wired to do. This leads to an important shift in perspective: a program is not something the computer _runs_ in a human sense. It’s something the CPU **steps through**, one instruction at a time.

Internally, the processor follows a simple repetitive cycle. It fetches the next instruction from memory, decodes what that instruction means, executes it, and then moves on. Over and over again, billions of times per second. There is no built-in concept of _“this instruction belongs to a function”_ or _“we are inside a loop.”_ The CPU doesn’t know it is executing a program at all. It only knows the address of the current instruction, and where to go next. If nothing ever interrupted this process, execution would simply move forward in a straight line. The program would run from top to bottom exactly once and then stop. What makes software powerful is that this forward motion can be redirected.

To understand how this works, it helps to think of memory as a massive numbered grid. Every location has an address and stores a value. Some of these values represent instructions. Others represent data such as numbers, text, flags, or internal state. But to the CPU, it is all just memory. Nothing in the hardware says _“this is code”_ or _“this is a variable.”_ What matters is how a value is interpreted at a given moment. When the instruction pointer refers to an address, the value there is treated as an instruction. When another address is accessed, its value may be treated as data. The distinction exists mostly in the minds of programmers and in the structure imposed by compilers. This is where high-level programming languages quietly disappear.

When you write a variable, a loop, a function, or an `if` statement, none of those things exist in the executable file that the CPU actually runs. The compiler translates all of that structure into a flat sequence of low-level instructions and memory accesses. By the time the program reaches the processor, the careful organization you created has been erased and replaced with raw mechanics. Seen this way, execution can feel unsettling at first. The program no longer looks like something a human would write. It looks like a machine-specific script for moving values around and occasionally changing where execution continues.

The ability to change where execution continues is what makes computation interesting. This is known as **control flow**. Instead of always moving forward, the CPU can be told to jump to another address. A jump instruction says, in effect: _“Set the instruction pointer to this location instead.”_ From this single idea, everything else emerges. Loops are just backward jumps. Conditionals are jumps that only happen when a condition is met. Functions are jumps to another region of code, followed later by a jump back. Once you see this clearly, writing programs stop feeling like elaborate logical structures and start feeling like carefully **{{< mathjax/inline>}}<span style="color: #0084a5;">choreographed instruction pointer movements.</span>{{< /mathjax/inline>}}**

This leads to the biggest mental shift. There is no hidden machinery making programs behave nicely. There is no invisible system that understands “functions,” “scopes,” or “returns.” **{{< mathjax/inline>}}<span style="color: #0084a5;">Every abstraction we rely on must ultimately be built out of sequential execution, memory reads and writes, and jumps.</span>{{< /mathjax/inline>}}** Nothing more. Understanding that a program is ultimately just a list of instructions, and that execution is nothing more than the CPU walking that list - sometimes stepping forward, sometimes jumping elsewhere - is the foundation for everything that follows.

Once you have that mental model, the questions change. Instead of asking, _“How does the computer understand functions?”_ you start asking, _“How do we fake functions using only jumps and memory?”_ And that’s where things get really interesting.

## How CPUs Move Through Code

---

As discussed in the previous section, once you strip away the abstractions, a CPU doesn’t “run a program” so much as it **moves through memory** or **steps through instruction**. Execution is not a vague process - it is a physical, mechanical progression from one memory address to the next. At the center of this process is a special register called the **instruction pointer**, sometimes known as the **program counter**. This register holds a single number: the memory address of the instruction that should be executed next. At every moment the CPU is active, it answers exactly one question:

> What instruction should I execute right now?

Most of the time, the processor behaves in the most boring way possible. It fetches the instruction at the current address, executes it, and then advances the instruction pointer so it points to the next instruction in memory. If one instruction sits at address `1000` and the next at `1004`, execution simply moves from `1000` to `1004` to `1008`, and so on.

From the CPU’s point of view, this is the “normal” case. There is no awareness of lines of code, blocks, or structure - only numbers increasing step by step. If programs only worked this way, they would be completely linear. There would be no loops, no decisions, and no reuse of logic.

What makes computation powerful is the ability to break this straight line. Certain instructions allow the program to directly overwrite the instruction pointer. These are **jumps**. A jump tells the CPU not to continue forward, but to resume execution at some other address. When this happens, execution does not “travel” there - it is instantly relocated. Once the jump is complete, normal sequential execution resumes from the new location.

From this single mechanism, nearly everything in programming emerges. Loops are backward jumps. Conditionals are jumps that only happen under certain conditions. Early exits are jumps that skip over code. Functions are jumps into other regions of memory.

To the hardware, there is no conceptual difference between these cases. They are all just decisions about which address comes next. When humans write low-level code, we often use labels instead of raw addresses. A label such as `loop_start` is simply a name for a particular memory location. During assembly, the label is replaced with an actual number. The CPU never sees the name. It only sees the resolved address.

This reinforces an important idea:

> Much of what makes code readable exists only before execution.

By the time the CPU becomes involved, clarity has been sacrificed for precision.

Another subtle but crucial fact follows from this: CPUs have no built-in concept of “returning.” When execution jumps somewhere, the processor does not remember where it came from. It simply starts executing at the new address. If a program wants to go back, it must explicitly store that information somewhere.

This is why function calls are not trivial at the machine level. The hardware provides no automatic way to say, “Go over there, do some work, and then come back here.” That behavior has to be constructed manually using memory and control flow.

<!-- And yet, despite this blindness, execution is perfectly reliable. The CPU never asks whether it is inside a function or a loop. It does not know whether an instruction is important. It only knows where the instruction pointer points and what the current instruction says to do. -->

By carefully arranging instructions and jumps, we create the illusion of structured, hierarchical programs. This leads to a quiet paradox at the heart of computing:

> Very simple rules, followed relentlessly, produce extremely complex systems.

Once you understand execution as nothing more than controlled movement through memory, many abstractions stop feeling mysterious. They become answers to a single question:

> How do we control where the CPU goes next?

## Functions Without Magic

---

At a high level, functions are presented as a way to organize code. You give a piece of logic a name, you call it, and execution continues as if that logic were a single instruction. Clean, elegant, and easy to reason about.

But at the hardware level, **functions do not exist.** The CPU has no idea what a function is. It does not know about parameters, return values, or call stacks. From its perspective, there is only memory and a pointer telling it where to execute next.

Functions are a convention - a disciplined way of arranging instructions and jumps so that a block of code can be reused without copying it everywhere. Imagine writing a program as one long sequence of instructions. The CPU starts at the top and walks forward. That is easy. Now imagine you want to reuse a piece of logic in two different places. You could duplicate the instructions, but that wastes space and makes maintenance painful. Or you could jump to a shared block of code and then return.

Jumping is easy. Returning is not.

Once execution leaves its original path, the CPU forgets where it came from. If we want a function to behave correctly, we must first record the return address somewhere, then jump to the function, and later jump back using that stored address.

<!-- When you write `do_something();`, it feels like a request. At the machine level, it is just a carefully engineered redirection of execution followed by a carefully engineered U-turn. -->

Because functions can be called from many places, return addresses cannot be hardcoded. They must be dynamic. They must depend on where the call originated. This is where compilers step in. When a compiler sees a function call, it generates code to store the return address in memory, jump to the function, and later restore execution using that stored value.

> Functions are not atomic. They are built out of memory operations and jumps.

This realization reveals the true cost of abstraction. Function calls have a real performance cost. Nested calls require careful bookkeeping. Reuse is always a tradeoff between clarity and overhead. Functions are engineered mechanisms for reuse, designed to look simple so humans can reason about programs.

## When Functions Disappear: Inlining

---

One way to avoid the complexity of calling and returning is to avoid jumping altogether. Instead of generating code that transfers execution elsewhere, the compiler can copy the body of a function directly into the place where it is called. This technique is known as **function inlining**.

From the CPU’s perspective, nothing special happens. There is no call, no return, and no change in control flow. Execution simply continues forward. The function vanishes. Inlining replaces modular structure with duplication. Each “call” becomes a copy of the function’s instructions. Parameters are substituted, local variables are adjusted, and the abstraction is erased. This is why inlining is often described as compile-time substitution.

<!-- Inlining removes the overhead of storing return addresses and manipulating the stack. It also gives the compiler a wider view of the code, enabling aggressive optimizations such as constant folding, dead code elimination, and instruction reordering. For these reasons, compilers favor inlining whenever it seems beneficial. -->

However, inlining has a physical cost: size. Excessive duplication increases binary size, puts pressure on instruction caches, and can eventually harm performance. Large or recursive functions cannot be inlined reliably. At some point, reuse requires sharing code.

When that happens, the compiler must return to outlining.

## Jumping Away and Coming Back: Outlining

---

With outlining, a function exists as a single block of code stored in one place. Every call jumps to that block and later returns. This solves the problem of duplication, but it reintroduces the problem of returning. A naive solution is to store the return address in a fixed memory location. This works only until functions call other functions. As soon as calls are nested, return addresses overwrite each other.

The problem is not storage itself. It is ordering. Multiple return addresses must coexist, and they must be restored in reverse order of creation. The most recent call must return first. This is a **Last-In, First-Out** pattern. It is the defining behavior of a **stack**.

At this point, function calls stop being a control-flow trick and become a data-structure problem. Without a structured way to store return addresses, reliable reuse is impossible.

## The Stack: Memory With Discipline

---

A stack is not special hardware. It is just a region of memory that we agree to use in a disciplined way. Alongside it, we maintain a **stack pointer** that tracks the current top. When a function is called, the return address is pushed onto the stack. When the function finishes, that address is popped off and used to resume execution. This simple mechanism solves the nested call problem automatically. Each function interacts only with the top of the stack. No function needs to know how deep it is or who called it.

<!-- Local simplicity produces global correctness. -->

Because function calls are so common, modern CPUs provide hardware support for stack management. Registers store the stack pointer. Special instructions combine jumping with pushing and popping. These features exist purely to accelerate this pattern.

Over time, the stack becomes more than a place for return addresses. Each call creates a **stack frame** that stores parameters, local variables, and saved register values. This is why recursion works: every call receives its own isolated slice of memory.

The stack is the quiet foundation of almost every program. You rarely see it in high-level code, yet it underlies nearly all execution. It exists because programs leave and return. Because execution has depth. Because memory must mirror that depth. And once you understand that, many of the mysteries of programming dissolve. What looks like magic is usually just careful bookkeeping built on top of very simple rules.

At the lowest level, programs are still just numbers. Execution is still just movement. And structure is still something we build - patiently and deliberately - on top of that.

## Inlining and Outlining: Choosing Wisely

---

After seeing how much machinery function outlining requires, it is tempting to conclude that inlining is always superior. No jumps. No stack manipulation. Fewer instructions. Faster execution.

At first glance, it seems obvious.

But that conclusion does not hold up for long.

Inlining and outlining solve different problems, and the “better” choice depends on constraints that are often invisible at the source-code level.

Inlining works by duplicating code. If a function is small and used in only a few places, this duplication barely matters. But if a function is large and widely used, inlining it everywhere can dramatically increase the size of the executable.

This matters more than it might seem.

Larger binaries consume more memory, take longer to load, and put pressure on the instruction cache. Once a program grows beyond what fits comfortably in cache, performance can degrade sharply. At that point, the cost of having more code outweighs the savings from avoiding function calls.

This is why compilers are cautious. They do not simply ask, “Is this function fast?” They ask, “Is it worth duplicating this code here?”

Outlining avoids duplication, but it introduces its own costs.

Every outlined function call requires saving a return address, jumping to another region of memory, and later returning. Arguments must be passed, results retrieved, and registers preserved. Individually, these operations are small. But inside tight loops or performance-critical paths, they accumulate.

When the function being called is very short, the overhead of calling it can exceed the cost of the work it performs. In those cases, outlining is technically correct - but inefficient.

Modern CPUs complicate this tradeoff further through caching.

Processors do not fetch instructions directly from main memory every time. They rely on fast, limited caches that store recently used code. Inlining increases code size, which can cause instruction cache misses. Outlining reduces code size, but it may place frequently called functions far from their callers, forcing new cache lines to be loaded during jumps.

As a result, two logically identical programs can behave very differently depending on how their code is laid out in memory.

This leads to one of the most unintuitive truths about performance:

> Layout matters as much as logic.

Another important factor is frequency.

Not all code runs equally often. Some functions sit on hot paths and execute millions of times per second. Others handle rare conditions and may never run at all. Inlining a tiny hot function can be a major win. Inlining a large, rarely used function is often wasteful.

Conversely, outlining cold code can improve overall performance by keeping hot paths compact and cache-friendly. This is why compilers and performance engineers think in terms of execution frequency, not just correctness.

There are also cases where inlining is impossible.

The most obvious example is recursion. When a function can call itself an unknown number of times, the compiler cannot inline it fully. There is no finite amount of code that can represent an unbounded call depth.

In such cases, outlining is not just better - it is essential. And once outlining is required, the stack and all its associated machinery return.

Modern compilers navigate these tradeoffs continuously. They evaluate function size, call frequency, optimization levels, target architectures, cache behavior, and even debugging constraints. They use heuristics and profiling data to make decisions that are good on average, not perfect in every case.

Languages may offer hints such as `inline` keywords or compiler attributes, but these are suggestions, not commands. The compiler ultimately decides.

Inlining and outlining are not about right or wrong. They are about tradeoffs.

They illustrate a deeper truth about systems programming:

> There is no free abstraction.

Every convenience carries a cost - whether in memory, execution time, complexity, or flexibility. Understanding this does not mean you should manually optimize everything. It means you gain intuition. Performance stops being mysterious and starts looking like the result of concrete design decisions.

Functions stop being “just functions.”

They become what they really are: carefully engineered compromises between clarity and control.

## What This Taught Me About Data Structures

---

Before diving into low-level execution, I used to think of data structures mainly as _abstract tools_: stacks, queues, trees, and graphs that existed mostly in textbooks and APIs. They felt like conceptual conveniences - useful, but somewhat detached from “real” computation.

Understanding how programs actually execute changed that perspective completely.

The call stack, in particular, stopped being just a diagram in lecture slides. It became a living structure that the CPU constantly manipulates. Every function call is a push. Every return is a pop. Every local variable lives inside a carefully allocated frame. What once looked like an academic idea is, in reality, one of the most heavily used data structures in computing.

This realization reframed how I think about abstraction.

High-level languages do not eliminate complexity. They reorganize it.

When you write:

```cpp
compute();
```

You are implicitly asking the system to:

- Store where execution currently is
- Allocate space for local variables
- Transfer control elsewhere
- Preserve registers
- Restore everything later

All of that behavior is coordinated through structured memory layouts that resemble classical data structures.

The same applies elsewhere:

- Heaps are not “just memory” - they are managed forests of blocks.
- Objects are layouts with metadata.
- Closures are packaged environments.
- Recursion is a stack-driven algorithm.

Once you see this, data structures stop being optional design choices. They become the language through which computation expresses itself.

This also explains why “choosing the right data structure” matters so much. It is not just about asymptotic complexity. It is about aligning your mental model with how machines actually organize information.

Good programmers are not just writing logic.

They are shaping memory.

## Closing Thoughts: No Magic - Just Careful jumps

---

At first glance, modern software feels almost supernatural.

You click a button. A neural network runs. A 3D world appears. Millions of lines of code cooperate flawlessly. Errors are caught. Windows redraw themselves. Data flows invisibly.

It feels like magic.

But under the hood, something surprisingly simple is happening.

The processor is jumping.

Over and over again.

It jumps to functions.
It jumps to error handlers.
It jumps to libraries.
It jumps back.
It jumps forward.
It jumps based on conditions.

Every abstraction - loops, functions, objects, exceptions, threads - is built on this foundation of controlled jumps and saved locations.

Nothing “flows” naturally.

Everything is redirected deliberately.

The call stack remembers where to go back.
The instruction pointer decides where to go next.
The compiler rearranges jumps for performance.
The linker stitches jumps across files.
The loader maps jumps into memory.

A modern application is not a continuous story.

It is a carefully choreographed dance of redirections.

Understanding this removes a lot of mystery.

Debugging becomes less intimidating.
Performance issues become more traceable.
Crashes become more explainable.
Optimizations become more meaningful.

You stop seeing code as static text.

You start seeing it as motion.

As trajectories through memory.

As paths taken and untaken.

This is why learning how programs execute matters - not just for systems engineers, but for anyone who writes code seriously.

It teaches humility.

Your elegant abstraction eventually becomes:

- An address
- A jump
- A stack frame
- A return

And somehow, from those pieces, entire worlds emerge.

Not through magic.

Through structure.

Through discipline.

Through carefully organized jumps.

## <!-- ## Program at the Lowest Level

At its core, a program is not an idea, a workflow, or a set of intentions. It’s not even _logic_ in the way we usually think about it. **{{< mathjax/inline>}}<span style="color: #0084a5;">At the lowest level, a program is just data - a long sequence of numbers stored in memory.</span>{{< /mathjax/inline>}}**

Those numbers are interpreted as **instructions**.

A CPU doesn’t see variables, functions, loops, or conditionals. It sees a stream of encoded commands like _“move this value,” “add these numbers,”_ or _“jump to that address.”_ Each instruction is small, mechanical, and boring on its own. There is no understanding of _why_ the instruction exists or what it contributes to the larger program. The CPU simply executes it because that’s what it’s wired to do.

This is an important shift in perspective:

> A program is not something the computer _runs_ in a human sense - it’s something the CPU **steps through**.

### One Instruction at a Time

The CPU follows a simple, repetitive cycle:

1. Fetch the next instruction from memory
2. Decode what that instruction means
3. Execute it
4. Move to the next instruction

That’s it.

There’s no built-in concept of _“this instruction belongs to a function”_ or _“we are inside a loop.”_ The CPU doesn’t know it’s executing a program at all. It just has a pointer to a memory address - the location of the current instruction - and after each step, that pointer usually moves forward to the next address.

If nothing ever interrupted this process, a program would just be a straight line of instructions, executed from top to bottom, exactly once.

### Memory Is Just a Giant Numbered Grid

To make sense of how this works, it helps to think of memory as a massive list of numbered boxes. Each box has an address - like a house number - and stores a value.

Some of those boxes contain:

- Instructions (the program itself)
- Data (numbers, text, flags, state)

The CPU doesn’t inherently care which is which. It’s all just memory. What matters is **how the value in a memory location is interpreted at a given moment**.

When the CPU fetches an instruction, it reads the value at the address stored in its instruction pointer. When it reads or writes data, it accesses other addresses. Nothing about memory says _“this is code”_ or _“this is a variable.”_ That distinction exists almost entirely in how the compiler and the programmer intend the memory to be used.

### Where High-Level Code Disappears

This is where high-level programming languages quietly vanish.

When you write:

- a variable
- a loop
- a function
- an `if` statement

none of those things exist in the executable file the CPU actually runs.

The compiler’s job is to translate all of that into a flat sequence of low-level instructions and memory accesses. By the time the CPU sees the program, the structure you carefully designed has been erased and replaced with raw mechanics.

This is why understanding execution at this level can feel unsettling at first. The program no longer looks like something a human would write. It looks like a machine-specific script for moving values around and occasionally changing where execution continues.

### Flow Control Is the Real Superpower

If a program were truly just a straight line, it wouldn’t be very useful. The key capability that makes computation interesting is **control flow** - the ability to _not_ execute the next instruction.

This is where instructions like **jumps** come in. A jump tells the CPU:

> “Instead of continuing forward, set the instruction pointer to this other address.”

With that single idea, everything else becomes possible:

- Loops are jumps backward
- Conditionals are jumps that only happen sometimes
- Functions are jumps to another region of code

Once you realize this, programs stop feeling like elaborate logical structures and start feeling like carefully **{{< mathjax/inline>}}<span style="color: #0084a5;">choreographed instruction pointer movements.</span>{{< /mathjax/inline>}}**

### The Big Mental Shift

The biggest realization for me was this:

> There is no hidden machinery making programs behave nicely.

**Every abstraction we rely on - functions, scopes, calls, returns - must be built out of:**

- **sequential execution**
- **memory reads and writes**
- **jumps**

Nothing more.

Understanding that a program is ultimately just a list of instructions, and that _execution_ is nothing more than the CPU walking that list - sometimes stepping forward, sometimes jumping elsewhere - is the foundation for everything that follows.

Once you have that mental model, the questions change.

Instead of asking:

> _“How does the computer understand functions?”_

you start asking:

> _“How do we fake functions using only jumps and memory?”_

And that’s where things get really interesting. -->

## <!-- ## Program at the Lowest Level

At its core, a program is not an idea, a workflow, or a set of intentions. It’s not even _logic_ in the way we usually think about it. **{{< mathjax/inline>}}<span style="color: #0084a5;">At the lowest level, a program is just data - a long sequence of numbers stored in memory.</span>{{< /mathjax/inline>}}** Those numbers are interpreted as **instructions**.

A CPU doesn’t see variables, functions, loops, or conditionals. It sees only a stream of encoded commands like _“move this value,” “add these numbers,”_ or _“jump to that address.”_ Each instruction is small, mechanical, and boring on its own. There is no understanding of _why_ the instruction exists or what it contributes to the larger program. The CPU simply executes it because that’s what it’s wired to do.

This leads to an important shift in perspective: a program is not something the computer _runs_ in a human sense. It’s something the CPU **steps through**, one instruction at a time.

Internally, the processor follows a simple repetitive cycle. It fetches the next instruction from memory, decodes what that instruction means, executes it, and then moves on. Over and over again, billions of times per second. There is no built-in concept of _“this instruction belongs to a function”_ or _“we are inside a loop.”_ The CPU doesn’t know it is executing a program at all. It only knows the address of the current instruction, and where to go next.

If nothing ever interrupted this process, execution would simply move forward in a straight line. The program would run from top to bottom exactly once and then stop. What makes software powerful is that this forward motion can be redirected.

To understand how this works, it helps to think of memory as a massive numbered grid. Every location has an address and stores a value. Some of these values represent instructions. Others represent data such as numbers, text, flags, or internal state. But to the CPU, it is all just memory. Nothing in the hardware says _“this is code”_ or _“this is a variable.”_

What matters is how a value is interpreted at a given moment. When the instruction pointer refers to an address, the value there is treated as an instruction. When another address is accessed, its value may be treated as data. The distinction exists mostly in the minds of programmers and in the structure imposed by compilers.

This is where high-level programming languages quietly disappear.

When you write a variable, a loop, a function, or an `if` statement, none of those things exist in the executable file that the CPU actually runs. The compiler translates all of that structure into a flat sequence of low-level instructions and memory accesses. By the time the program reaches the processor, the careful organization you created has been erased and replaced with raw mechanics.

Seen this way, execution can feel unsettling at first. The program no longer looks like something a human would write. It looks like a machine-specific script for moving values around and occasionally changing where execution continues.

The ability to change where execution continues is what makes computation interesting. This is known as **control flow**. Instead of always moving forward, the CPU can be told to jump to another address. A jump instruction says, in effect: _“Set the instruction pointer to this location instead.”_

From this single idea, everything else emerges. Loops are just backward jumps. Conditionals are jumps that only happen when a condition is met. Functions are jumps to another region of code, followed later by a jump back.

Once you see this clearly, programs stop feeling like elaborate logical structures and start feeling like carefully **{{< mathjax/inline>}}<span style="color: #0084a5;">choreographed instruction pointer movements.</span>{{< /mathjax/inline>}}**

This leads to the biggest mental shift.

There is no hidden machinery making programs behave nicely. There is no invisible system that understands “functions,” “scopes,” or “returns.” Every abstraction we rely on must ultimately be built out of sequential execution, memory reads and writes, and jumps. Nothing more.

Understanding that a program is ultimately just a list of instructions, and that execution is nothing more than the CPU walking that list - sometimes stepping forward, sometimes jumping elsewhere - is the foundation for everything that follows.

Once you have that mental model, the questions change.

Instead of asking, _“How does the computer understand functions?”_ you start asking, _“How do we fake functions using only jumps and memory?”_

And that’s where things get really interesting.

## How CPUs Move Through Code

---

Once you strip away the abstractions, a CPU doesn’t “run a program” so much as it **moves through memory**. Execution is not a vague process - it’s a physical, mechanical progression from one memory address to the next.

At the center of this process is a special register inside the CPU called the **instruction pointer** (sometimes called the **program counter**). This register holds a single number: the memory address of the instruction that should be executed next.

Every moment the CPU is active, that register answers exactly one question:

> What instruction should I execute right now?

### The Default Behavior: Just Keep Going

Most of the time, the CPU behaves in the most boring way possible.

It fetches the instruction located at the address in the instruction pointer, executes it, and then increments the instruction pointer so it points to the next instruction in memory.

If instruction A is stored at address `1000` and instruction B is right after it at `1004`, execution looks like this:

- Execute instruction at `1000`
- Instruction pointer becomes `1004`
- Execute instruction at `1004`
- Instruction pointer becomes `1008`

From the CPU’s point of view, this is the “normal” case. There is no awareness of lines of code, blocks, or structure - just addresses increasing one step at a time.

If programs only worked this way, they would be completely linear.

No loops.  
No decisions.  
No reuse of code.

### Breaking the Line: Jumps

The real power comes from instructions that change the instruction pointer directly.

A **jump** instruction tells the CPU:

> “Do not continue to the next instruction. Instead, go execute the instruction at this other address.”

This is a fundamental idea. The CPU doesn’t _visit_ the target code - it **teleports execution** there by overwriting the instruction pointer.

Once that happens, sequential execution resumes from the new location.

This one mechanism enables nearly everything we associate with programming:

- Loops are jumps backward
- Conditionals are jumps that only occur if a condition is met
- Early exits are jumps that skip over code
- Functions are jumps into another region of memory

From the hardware’s perspective, there’s no difference between these concepts. They’re all just ways of deciding which address comes next.

### Labels Are for Humans, Not CPUs

When we write assembly code, we usually don’t work with raw memory addresses. Instead, we use **labels** - names that represent locations in code.

For example, a label like `loop_start` might represent a specific address in memory. When the program is compiled, the assembler replaces that label with an actual numeric address.

The CPU never sees the label. It only sees the resolved number.

This reinforces an important idea:

> Much of what makes code readable exists only before execution.

By the time the CPU gets involved, clarity has been sacrificed for precision.

### There Is No “Going Back” Built In

One subtle but crucial detail is this:

> CPUs have no inherent concept of returning.

When a jump occurs, the CPU does not remember where it came from. It simply starts executing from the new address. If you want to go back, you must explicitly tell the CPU where “back” is.

This is why function calls are not trivial at the machine level. The hardware provides no automatic way to say:

> “Jump over there, do some work, then come back here.”

That behavior has to be constructed manually using memory and control flow - which is exactly the problem function outlining must solve.

### Execution Is Blind but Reliable

The CPU never asks:

- “Am I inside a function?”
- “Is this a loop?”
- “Is this instruction important?”

It doesn’t know what came before, and it doesn’t care what comes after. It only knows:

- Where the instruction pointer points
- What the current instruction tells it to do

And yet, by carefully arranging instructions and jumps, we can create the illusion of structured, hierarchical programs with well-defined behavior.

This is the paradox at the heart of computing:

> Very simple rules, followed relentlessly, produce extremely complex systems.

Understanding how CPUs move through code is the key to dissolving that paradox. Once you see execution as nothing more than controlled movement through memory, the rest of the abstractions - functions, stacks, and data structures - stop feeling mysterious.

They become solutions to a very specific problem:

> How do we control where the CPU goes next?

And that question leads directly to the idea of function calls, returns, and eventually - the stack.

## Functions: Reusing Code Without Magic

---

At a high level, functions are sold to us as a way to organize code. You give a piece of logic a name, you call it when you need it, and execution continues as if the function were a single instruction. Clean, elegant, and easy to reason about. But that explanation hides something important: **functions don’t exist at the hardware level.**

The CPU has no idea what a function is. It doesn’t know about parameters, return values, or call stacks. From its perspective, there is only code stored in memory and a pointer telling it where to execute next.

**The Great Illusion**

If functions aren’t "real" to the CPU, what are they? They are a **convention** - a disciplined way of arranging instructions and jumps so that a chunk of code can be reused without copying it everywhere.

### Why Reuse Is Harder Than It Looks

Imagine writing a program as one long sequence of instructions. The CPU starts at the top and walks forward. That’s easy. Now imagine you want to reuse a piece of logic in two different places. You have two options:

1. **Copy and paste the instructions:** This works, but it's wasteful. Your program grows larger, and updates become a maintenance nightmare.
2. **Jump to the same code and come back:** This sounds simple, but it introduces a subtle problem: _Once you jump away, how do you know where to return?_

Humans understand context. CPUs don’t. A jump wipes out the current execution path unless we explicitly preserve it.

### The Anatomy of a Call

When we write `do_something();`, it feels like a request: _"Please run this function and then resume."_ At the machine level, there is only a redirection of execution. To make a function behave correctly, two things must happen:

- **The Leap:** Execution must jump to the function’s code.
- **The U-Turn:** The function must later jump back to the correct location.

A function is useful precisely because it can be called from multiple places. This means there is no single “return location.”

- If called from inside a **loop**, it must return to the loop.
- If called from **anywhere else**, it must return there instead.

Hardcoding a return address into the function doesn’t work. The return address must be **dynamic** - it must depend on exactly where the call originated.

### What the Compiler Actually Does

Because the CPU won’t manage this automatically, the compiler does the heavy lifting. When it sees a function call, it generates a sequence of instructions to:

1. **Store** the location where execution should resume.
2. **Jump** to the function’s code.
3. **Restore** execution afterward using that stored address.

> **Key Insight:** Functions are not atomic. They are built out of simpler parts: memory writes, memory reads, and jumps.

### Why This Matters

Once you see functions this way, the "cost" of abstraction becomes clear:

- **Performance:** Calling and returning from a function has a real (though small) CPU cost.
- **Complexity:** Nested calls require careful bookkeeping.
- **Tradeoffs:** Reuse is a balance between clean code and execution overhead.

Functions are carefully engineered mechanisms for reuse, designed to look simple so humans can reason about programs more easily. The next two sections discuss about ways of function implementation.

## Function Inlining: When Functions Disappear

---

The simplest way to implement a function is surprisingly blunt: **don’t jump anywhere at all.**

Instead of generating code that transfers execution to another part of memory, the compiler can take the body of a function and copy it directly into the place where the function is called. This technique is known as **function inlining**.

From the CPU’s perspective, nothing special happens. There is no call, no return, and no change in control flow. Execution just continues forward, instruction by instruction. In a sense, the function vanishes.

### What Inlining Really Means

Suppose you write a small function for a simple calculation. At the source-code level, it looks modular and reusable. But when the compiler inlines it, that abstraction is erased.

- **Duplication:** Every time the function is “called,” its instructions are duplicated into the caller’s code.
- **Replacement:** Parameters are replaced with appropriate values.
- **Adjustment:** Local variables are shifted so they don't conflict with the caller's variables.

By the time the program reaches the CPU, there is no trace of the function - only a longer, continuous sequence of instructions. This is why inlining is often called **compile-time substitution**.

### Why Compilers Love Inlining

Inlining sidesteps the mechanical overhead of a standard function call:

1. There is no return address to store.
2. There is no need for a stack.
3. The instruction pointer simply keeps moving forward.

Beyond saving a few CPU cycles, inlining allows the compiler to see the "big picture." Once the function is merged into the caller, the compiler can perform advanced optimizations:

- **Constant Folding:** If the inputs are known, the compiler can pre-calculate the result.
- **Dead Code Elimination:** Removing parts of the function that aren't actually needed in that specific context.
- **Instruction Reordering:** Mixing the function’s logic with the caller's logic for better pipeline efficiency.

### The Illusion of Modularity

Inlining preserves the **illusion of modularity** for the programmer while discarding it for the machine. You write clean, named behaviors for human readability, but the compiler quietly flattens that structure to favor machine performance.

> **The Reality:** How we write code is optimized for human understanding; how code runs is optimized for the hardware. The compiler is the bridge between those two conflicting goals.

### When Inlining Stops Being a Good Idea

Inlining isn’t a "silver bullet." Because it relies on duplication, it comes with a physical cost: **Size.**

| Factor                | Impact of Excessive Inlining                                                     |
| --------------------- | -------------------------------------------------------------------------------- |
| **Binary Size**       | The final executable grows significantly larger.                                 |
| **Instruction Cache** | A larger "hot path" can overflow the CPU's cache, leading to slower performance. |
| **Complexity**        | Extremely large functions become harder for the compiler to optimize further.    |

Compilers use **heuristics** (rules of thumb) to decide when to inline. They weigh the size of the function against how often it is called and the current optimization settings (e.g., `-O2` vs `-Os`).

### The Important Takeaway

Inlining proves that **functions are not required for execution.** They are a convenience for humans. When a compiler can erase them without a performance penalty, it usually does.

However, inlining only works when the program structure is known and finite. When functions are too large, shared across different programs, or **recursive**, the illusion breaks down. In those cases, we must use **function outlining**: keeping one copy of the code and jumping to it.

This forces us back to the original hard problem: _How do you leave one place in a program and reliably come back?_

## Function Outlining: Jumping and Coming Back

---

If function inlining works by pretending functions don’t exist, **function outlining** does the opposite. It embraces the idea that a function is a real, separate block of code stored somewhere else in memory - and that execution must physically jump to it at runtime.

Instead of copying the function body everywhere it’s used, the compiler generates the function’s code **once**, places it in its own region of memory, and inserts jump logic at every call site. This immediately solves one problem: **code duplication.** But it introduces a much harder one.

### Jumping Is Easy. Returning Is Not.

Getting into a function is trivial. The CPU already knows how to jump. When execution reaches a function call, the program simply jumps to the memory address where the function begins.

**The real challenge is what happens next.**

Once the function finishes executing, how does the CPU know where to go? There is no built-in notion of a “caller.” The CPU does not remember where it jumped from. Without extra work, once execution leaves the original code path, it is effectively lost.

> **Crucial Concept:** Returning is not something the hardware does for you. It is something you must **engineer**.

### The Tempting but Broken Solution

A natural first idea is to **hardcode** a return location. At the end of the function, you add a jump back to the instruction immediately following the call.

- **The Flaw:** Functions are valuable because they are reusable.
- **The Result:** If a function is called from two different places, a hardcoded return address can only be correct for one of them. The function would always jump back to the same spot, regardless of who called it.

### Making the Return Address Dynamic

To fix this, the return address must depend on where the call originated. We can store the return address in memory _before_ jumping. The function then reads this stored address to perform its U-turn.

This works - **as long as functions never call other functions.**

### The Nested Call Problem

As soon as functions call other functions, the "single storage" solution collapses.

1. **Main Program** stores its return address (Point A) and jumps to **Function 1**.
2. **Function 1** stores its return address (Point B) and jumps to **Function 2**.
3. **The Crash:** Both calls use the same memory location. The second call **overwrites** the first.

When Function 2 returns to Point B, it works. But when Function 1 tries to return, Point A is gone. Execution jumps to the wrong place or loops forever.

### Why This Is a Data Structure Problem

The issue isn't storing a return address; it’s that multiple return addresses must coexist. They must be restored in a specific order: the most recent call must return first.

This requirement follows a strict pattern: **Last-In, First-Out (LIFO).**

This is the definition of a **Stack**. Function outlining stops being a control-flow trick and becomes a data-structure problem. Without a structured way to store these addresses, reliable function calls are impossible.

### What the Compiler Is Really Doing

When a compiler chooses outlining, it injects "hidden machinery" into your program:

- Code to **save** return addresses.
- Code to **restore** them later.
- A disciplined structure to prevent **overwriting**.
- Rules that guarantee correctness across **nested calls**.

This is why function calls are not “cheap.” They involve real memory operations and physical constraints.

### The Big Shift in Perspective

Function outlining forces us to confront a key idea: **Functions are not fundamental. The stack is.**

Once you allow outlined functions, you are implicitly agreeing to maintain a stack of execution contexts. Solving this unlocks the true power of modern programming:

- Deep call chains
- Modular libraries
- **Recursion**

All of it rests on one idea: **preserving the path of execution while temporarily leaving it.**

### The Next Step: The Stack

This leads to a final problem: once you allow functions to call other functions - or allow **recursion** - you need a way to store not just one return address, but many. Solving that problem leads directly to one of the most important data structures in computing: **the stack.**

## The Stack: A Data Structure That Makes It All Work

---

By the time we reach this point, the problem is clear: functions can’t return reliably unless each call remembers where it came from. A single return address isn’t enough because functions can call other functions in a chain.

We need a way to store many return addresses, keep them isolated, and retrieve them in the correct order. This requirement leads directly to one of the most important data structures in computing: **the stack**.

### Why a Stack Is the Right Shape

A stack is defined by two simple rules:

1. You can only add a new element to the **top**.
2. You can only remove the **most recently added** element.

This “last in, first out” (**LIFO**) behavior matches function calls perfectly. The most recent function call must return first. Once it returns, the function that called it becomes the most recent one.

### The Stack Is Just Memory

Despite how central it is, the stack is not "special" memory. It is just a region of normal RAM that we agree to use in a disciplined way. To make it work, we need:

- **A block of memory** reserved for the stack.
- **A stack pointer:** A dedicated register or variable that stores the memory address of the current "top" of the stack.

### Growing and Shrinking the Stack

- **Push:** When a function is called, the program moves the stack pointer and stores the return address there.
- **Pop:** When a function finishes, it reads the value at the stack pointer and moves the pointer back.

This mechanical simplicity is what makes the system so fast and reliable.

### Nested Calls, Naturally Handled

This is where the elegance of the stack shines. As execution dives deeper, the stack grows. As functions finish, the stack shrinks.

- Each function only interacts with the **top** of the stack.
- No function needs to know its depth or the identity of its caller.
- This **local simplicity** is what makes **global correctness** possible.

### From Software to Hardware

In theory, a stack could be managed entirely by software. In practice, that would be too slow for the most common operation in computing. To optimize this, CPUs provide dedicated hardware support:

| Hardware Feature           | Purpose                                                                      |
| -------------------------- | ---------------------------------------------------------------------------- |
| **Stack Pointer Register** | A high-speed storage location for the "top" address (e.g., `RSP` or `ESP`).  |
| **Push/Pop Instructions**  | Single commands that move the pointer and write/read memory simultaneously.  |
| **Call/Ret Instructions**  | Specialized jumps that automatically handle the return address on the stack. |

### More Than Return Addresses

While return addresses are the priority, the stack quickly becomes the "scratchpad" for the entire program. Each function call gets its own slice of the stack - a **Stack Frame** - which stores:

- **Function parameters**
- **Local variables**
- **Saved register state** (so the function doesn't break the caller's data)

This is why recursion works: each call to the same function gets a _different_ frame on the stack, so they never overwrite each other’s local variables.

### The Quiet Foundation

The most remarkable thing about the stack is its invisibility. You don’t declare it or manage it in your high-level code, yet it is the backbone of almost every program ever written.

Data structures like the stack aren't just abstract concepts for interviews; they are practical, inevitable solutions to the physical constraints of execution. The stack exists because programs leave and return, because execution has depth, and because memory must mirror that depth. -->
