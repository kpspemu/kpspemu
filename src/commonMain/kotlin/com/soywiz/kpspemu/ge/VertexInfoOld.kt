package com.soywiz.kpspemu.ge

/*
export class VertexInfo {
	// Calculated
	weightOffset:number = 0;
	textureOffset:number = 0;
	colorOffset:number = 0;
	normalOffset:number = 0;
	positionOffset:number = 0;
	textureComponentsCount:number = 0;
	align: number;
	size: number;

	// Extra
	value: number = -1;
	reversedNormal: boolean;
	address: number;

	// Attributes
	weight: NumericEnum;
	texture: NumericEnum;
	color: ColorEnum;
	normal: NumericEnum;
	position: NumericEnum;

	// Vertex Type
	index: IndexEnum;
	weightCount: number;
	morphingVertexCount: number;
	transform2D: boolean;
	weightSize:number;
	colorSize:number;
	textureSize:number;
	positionSize:number;
	normalSize:number;

	clone() {
		return new VertexInfo().copyFrom(this);
	}

	copyFrom(that:VertexInfo) {
		this.weightOffset = that.weightOffset;
		this.textureOffset = that.textureOffset;
		this.colorOffset = that.colorOffset;
		this.normalOffset = that.normalOffset;
		this.positionOffset = that.positionOffset;
		this.textureComponentsCount = that.textureComponentsCount;
		this.value = that.value;
		this.size = that.size;
		this.reversedNormal = that.reversedNormal;
		this.address = that.address;
		this.texture = that.texture;
		this.color = that.color;
		this.normal = that.normal;
		this.position = that.position;
		this.weight = that.weight;
		this.index = that.index;
		this.weightCount = that.weightCount;
		this.morphingVertexCount = that.morphingVertexCount
		this.transform2D = that.transform2D;
		this.weightSize = that.weightSize;
		this.colorSize = that.colorSize;
		this.textureSize = that.textureSize
		this.positionSize = that.positionSize;
		this.normalSize = that.normalSize;
		this.align = that.align;
		return this;
	}

	setState(state:GpuState) {
		let vstate = state.vertex;
		this.address = vstate.address;

		if ((this.value != vstate.value) || (this.textureComponentsCount != state.texture.textureComponentsCount) || (this.reversedNormal != vstate.reversedNormal)) {
			this.textureComponentsCount = state.texture.textureComponentsCount;
			this.reversedNormal = vstate.reversedNormal;
			this.value = vstate.value;
			this.texture = vstate.texture;
			this.color = vstate.color;
			this.normal = vstate.normal;
			this.position = vstate.position;
			this.weight = vstate.weight;
			this.index = vstate.index;
			this.weightCount = vstate.weightCount;
			this.morphingVertexCount = vstate.morphingVertexCount;
			this.transform2D = vstate.transform2D;

			this.updateSizeAndPositions();
		}

		return this;
	}

	updateSizeAndPositions() {
		this.weightSize = VertexInfo.NumericEnumSizes[this.weight];
		this.colorSize = VertexInfo.ColorEnumSizes[this.color];
		this.textureSize = VertexInfo.NumericEnumSizes[this.texture];
		this.positionSize = VertexInfo.NumericEnumSizes[this.position];
		this.normalSize = VertexInfo.NumericEnumSizes[this.normal];

		this.size = 0;
		this.size = MathUtils.nextAligned(this.size, this.weightSize);
		this.weightOffset = this.size;
		this.size += this.realWeightCount * this.weightSize;

		this.size = MathUtils.nextAligned(this.size, this.textureSize);
		this.textureOffset = this.size;
		this.size += this.textureComponentsCount * this.textureSize;

		this.size = MathUtils.nextAligned(this.size, this.colorSize);
		this.colorOffset = this.size;
		this.size += 1 * this.colorSize;

		this.size = MathUtils.nextAligned(this.size, this.normalSize);
		this.normalOffset = this.size;
		this.size += 3 * this.normalSize;

		this.size = MathUtils.nextAligned(this.size, this.positionSize);
		this.positionOffset = this.size;
		this.size += 3 * this.positionSize;

		this.align = Math.max(this.weightSize, this.colorSize, this.textureSize, this.positionSize, this.normalSize);
		this.size = MathUtils.nextAligned(this.size, this.align);
	}

	oneWeightOffset(n:number) {
		return this.weightOffset + this.weightSize * n;
	}

	private static NumericEnumSizes = [0, 1, 2, 4];
	private static ColorEnumSizes = [0, 0, 0, 0, 2, 2, 2, 4];

	get realWeightCount() { return this.hasWeight ? (this.weightCount + 1) : 0; }
	get realMorphingVertexCount() { return this.morphingVertexCount + 1; }
	get hasTexture() { return this.texture != NumericEnum.Void; }
	get hasColor() { return this.color != ColorEnum.Void; }
	get hasNormal() { return this.normal != NumericEnum.Void; }
	get hasPosition() { return this.position != NumericEnum.Void; }
	get hasWeight() { return this.weight != NumericEnum.Void; }
	get hasIndex() { return this.index != IndexEnum.Void; }
	get positionComponents() { return 3; }
	get normalComponents() { return 3; }
	get colorComponents() { return 4; }
	get textureComponents() { return this.textureComponentsCount; }
	get hash() { return this.value + (this.textureComponentsCount * Math.pow(2, 24)); }

	read(memory: Memory, count: number) {
		//console.log('read vertices ' + count);
		var vertices:any[] = [];
		for (var n = 0; n < count; n++) vertices.push(this.readOne(memory));
		return vertices;
	}

	private readOne(memory: Memory) {
		var address = this.address;
		var vertex: any = {};

		//console.log(vertex);
		this.address += this.size;

		return vertex;
	}

	toString() {
		return 'VertexInfo(' + JSON.stringify({
			address: this.address,
			texture: this.texture,
			color: this.color,
			normal: this.normal,
			position: this.position,
			weight: this.weight,
			index: this.index,
			realWeightCount: this.realWeightCount,
			morphingVertexCount: this.morphingVertexCount,
			transform2D: this.transform2D,
		}) + ')';
	}
}

class GpuState {
	val data = IntArray(512);
	val dataf = Float32Array (this.data.buffer);
	fun copyFrom(that: GpuState) {
		return this.writeData(that.data); }

	fun writeData(data: Uint32Array) {
		this.data.set(data); return this;
	}

	fun readData(): IntArray {
		return ArrayBufferUtils.cloneUint32Array(this.data);
	}
}

*/

